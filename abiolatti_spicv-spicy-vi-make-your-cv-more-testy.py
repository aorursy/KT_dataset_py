import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import colors

from sklearn import metrics, datasets, model_selection, linear_model, calibration

import xgboost, lightgbm

from tqdm import tqdm



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import backend as K
AUTO = tf.data.experimental.AUTOTUNE



def read_labeled_tfrecord(example):

    tfrec_format = {

        'image'                        : tf.io.FixedLenFeature([], tf.string),

        'image_name'                   : tf.io.FixedLenFeature([], tf.string),

        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),

        'sex'                          : tf.io.FixedLenFeature([], tf.int64),

        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),

        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),

        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),

        'target'                       : tf.io.FixedLenFeature([], tf.int64)

    }           

    example = tf.io.parse_single_example(example, tfrec_format)

    return example['image'], example['target']





def read_unlabeled_tfrecord(example):

    tfrec_format = {

        'image'                        : tf.io.FixedLenFeature([], tf.string),

        'sex'                          : tf.io.FixedLenFeature([], tf.int64),

        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),

        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),

        'image_name'                   : tf.io.FixedLenFeature([], tf.string),

    }

    example = tf.io.parse_single_example(example, tfrec_format)

    meta = tf.cast(tf.stack([example['sex'], example['age_approx']], 0), tf.float32)

    return example['image'], meta, example['image_name']



 

def prepare_image(img, dim=256):    

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.cast(img, tf.float32) / 255.0                      

    img = tf.reshape(img, [dim,dim, 3])

            

    return img





def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 

         for filename in filenames]

    return np.sum(n)
train_files = tf.io.gfile.glob("../input/melanoma-256x256/train*.tfrec")

test_files = tf.io.gfile.glob("../input/melanoma-256x256/test*.tfrec")
oof = pd.read_csv("../input/triple-stratified-kfold-with-tfrecords/oof.csv") # the kernel out of fold prediction

base_train_data = pd.read_csv("../input/melanoma-256x256/train.csv")
ds = tf.data.TFRecordDataset(train_files).map(read_unlabeled_tfrecord, num_parallel_calls=AUTO)

ds = ds.map(lambda i, m, n: (prepare_image(i), m, n), num_parallel_calls=AUTO)

ds = ds.batch(128)

ds = ds.map(lambda i, m, n: (K.concatenate([m, K.reshape(K.mean(i, [1, 2, 3]), [-1, 1]), K.mean(i, [1, 2]), K.std(i, [1, 2])], axis=1), n), num_parallel_calls=AUTO)

ds = ds.prefetch(AUTO)

data_train = np.array([i for i, n in tqdm(ds.unbatch())])

name_train = np.array([n.numpy().decode('utf-8') for i, n in tqdm(ds.unbatch())])



ds = tf.data.TFRecordDataset(test_files).map(read_unlabeled_tfrecord, num_parallel_calls=AUTO)

ds = ds.map(lambda i, m, n: (prepare_image(i), m, n), num_parallel_calls=AUTO)

ds = ds.batch(128)

ds = ds.map(lambda i, m, n: (K.concatenate([m, K.reshape(K.mean(i, [1, 2, 3]), [-1, 1]), K.mean(i, [1, 2]), K.std(i, [1, 2])]), n), num_parallel_calls=AUTO)

ds = ds.prefetch(AUTO)

data_test = np.array([i for i, n in tqdm(ds.unbatch())])

name_test = np.array([n.numpy().decode('utf-8') for i, n in tqdm(ds.unbatch())])
y = np.concatenate([np.zeros(data_train.shape[0]), np.ones(data_test.shape[0])], 0)

x = np.concatenate([data_train, data_test])

name = np.concatenate([name_train, name_test])
# train-test data distribution based on features (0: sex, 1: age, 2: img-avg, 3: img-std, 4..6: cw img-avg, 7..9: cw img-std)

i, j = 2, 3

bins = 50

plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)

c0, a0, b0, _ = plt.hist2d(x[y == 0, i], x[y == 0, j], bins=bins, density=True)

plt.pcolormesh(a0, b0, np.log(0 + c0).T);

plt.title("Training - Log density"); plt.xlabel("Color avg"); plt.ylabel("Color std")

plt.subplot(1, 3, 2)

plt.title("Test - Log density"); plt.xlabel("Color avg"); plt.ylabel("Color std")

c1, a1, b1, _ = plt.hist2d(x[y == 1, i], x[y == 1, j], bins=bins, density=True)

plt.pcolormesh(a1, b1, np.log(0 + c1).T);

plt.subplot(1, 3, 3)

plt.title("Abs diff"); plt.xlabel("Color avg"); plt.ylabel("Color std")

plt.pcolormesh(a0, b0, np.abs(c0 - c1).T);
idxT, idxV = list(model_selection.KFold(5, shuffle=True).split(x, y))[0]

model = xgboost.XGBClassifier()

model.fit(x[idxT], y[idxT], eval_set=[(x[idxT], y[idxT]), (x[idxV], y[idxV])], eval_metric='auc', verbose=0)

plt.plot(model.evals_result_['validation_0']['auc'], color='tab:blue')

plt.plot(model.evals_result_['validation_1']['auc'], color='tab:blue', linestyle=":")

plt.xlabel('n_estimators')

plt.ylabel('AUC');
calibrated_model = calibration.CalibratedClassifierCV(model, cv='prefit')

xcT, xcV, ycT, ycV = model_selection.train_test_split(x[idxV], y[idxV])

calibrated_model.fit(xcT, ycT)

plt.plot(*calibration.calibration_curve(ycV, model.predict_proba(xcV)[:, 1], n_bins=10)[::-1], color='tab:blue', label='XGB')

plt.plot(*calibration.calibration_curve(ycV, calibrated_model.predict_proba(xcV)[:, 1], n_bins=10)[::-1], color='tab:red', label='calibrated')

plt.plot(plt.xlim(), plt.ylim(), color='gray', linestyle='--', label='perfect')

plt.xlabel('Estimated probability'); plt.ylabel('True probability');

plt.legend();
NREPS = 10

NFOLDS = 10

df_all = []

models = NFOLDS * [xgboost.XGBClassifier()]

for rep in range(NREPS):

    pid = base_train_data[['patient_id']].drop_duplicates().reset_index(drop=True)

    folds = list(model_selection.KFold(NFOLDS, shuffle=True).split(pid))

    df_rep = []

    for fold, (pidIdxT, pidIdxV) in enumerate(folds):

        pidT, pidV = pid.iloc[pidIdxT]['patient_id'], pid.iloc[pidIdxV]['patient_id']

        inT = base_train_data[base_train_data['patient_id'].isin(pidT)]['image_name']

        inV = base_train_data[base_train_data['patient_id'].isin(pidV)]['image_name']

        idxT = (pd.Series(name).isin(inT) | y == 1)

        idxV = (pd.Series(name).isin(inV))

        model = models[fold]

        model.fit(x[idxT], y[idxT])

        # calibration should go here

        pred = model.predict_proba(x[idxV])[:, 1]

        testiness = pred / (1 - pred) * np.mean(y[idxT] == 0) / np.mean(y[idxT] == 1)

        df = pd.DataFrame({'rep': rep, 'image_name': name[idxV], 'is_test': y[idxV], 'is_test_pred': pred, 'testiness': testiness})

        df_rep.append(df)

        pass

    df_rep = pd.concat(df_rep, axis=0)

    df = pd.merge(df_rep, oof, on='image_name')

    score = metrics.roc_auc_score(df.target, df.pred)

    corrected_score = metrics.roc_auc_score(df.target, df.pred, sample_weight=df.testiness)

    print("Rep%2d \t\tAUC  = %.6f \tcAUC = %.6f" % (rep, score, corrected_score))

    df_all.append(df_rep)

    pass

df_all = pd.concat(df_all, axis=0)

df = pd.merge(df_all, oof, on='image_name').groupby('image_name').mean()

score = metrics.roc_auc_score(df.target, df.pred)

corrected_score = metrics.roc_auc_score(df.target, df.pred, sample_weight=df.testiness)

print("Overall \tAUC  = %.6f \tcAUC = %.6f" % (score, corrected_score))
XX = pd.DataFrame({'y': y, 'x2': x[:, 2], 'x3': x[:, 3], 'image_name': name}).merge(df, on='image_name')

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)

c0, a0, b0, _ = plt.hist2d(XX.x2[XX.y == 0], XX.x3[XX.y == 0], 50, density=True);

plt.subplot(1, 3, 2)

c1, a1, b1, _ = plt.hist2d(XX.x2[XX.y == 0], XX.x3[XX.y == 0], 50, density=True, weights=XX.testiness[XX.y == 0]);

plt.subplot(1, 3, 3)

ctest, atest, btest, _ = plt.hist2d(x[y == 1, 2], x[y == 1, 3], 50, density=True);

kl_pre = keras.losses.kld((c0 / c0.sum()).reshape(-1), (ctest / ctest.sum()).reshape(-1)).numpy()

kl_post = keras.losses.kld((c1 / c1.sum()).reshape(-1), (ctest / ctest.sum()).reshape(-1)).numpy()

print("KL divergence pre  = %.4f" % kl_pre)

print("KL divergence post = %.4f" % kl_post)

print("Difference = %.4f%%" % (100 * (kl_post - kl_pre) / kl_pre))
df_all[['image_name', 'testiness']].groupby('image_name').mean().to_csv("../working/testiness.csv")