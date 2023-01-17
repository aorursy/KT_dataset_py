!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index

!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index
import os

import datetime

from functools import lru_cache

import cv2

import pydicom

import pandas as pd

import numpy as np 

import tensorflow as tf 

import matplotlib.pyplot as plt 

import random

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import MinMaxScaler

from tensorflow_addons.optimizers import RectifiedAdam

from tensorflow.keras import Model

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tensorflow.keras.optimizers import Nadam

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

from colorama import Fore, Back, Style



def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



seed_everything(42)

ROOT = "../input/osic-pulmonary-fibrosis-progression/"
config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)
train = pd.read_csv(os.path.join(ROOT, 'train.csv'))

test = pd.read_csv(os.path.join(ROOT, 'test.csv'))
def get_agss_vector(df):

    

    """agss = age, gender, smokingstatus"""

    

    normalized_age = [(df.Age.values[0] - 30) / 30] 



    gender = [0 if df.Sex.values[0] == 'male' else 1]

    

    if df.SmokingStatus.values[0] == 'Never smoked':

        smoking_status = [0, 0]

    elif df.SmokingStatus.values[0] == 'Ex-smoker':

        smoking_status = [1, 1]

    elif df.SmokingStatus.values[0] == 'Currently smokes':

        smoking_status = [0, 1]

    else:

        smoking_status = [1, 0]



    vector = normalized_age + gender + smoking_status

    return np.array(vector)
def sample_best_fit_line_weeks_vs_fvc():

    

    patient = train.Patient.sample().iloc[0]

    sub = train.loc[train.Patient == patient, :]

    fvc = sub.FVC.values

    weeks = sub.Weeks.values

    vals = np.c_[weeks, np.ones(len(weeks))]  # column-wise stack

    

    # see example https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

    m, c = np.linalg.lstsq(vals, fvc, rcond=-1)[0]

    

    print(f"Patient number: {patient}")

    print("FVC", fvc)

    print("Weeks", weeks)

    print(vals)

    print(f"gradient: {m:.2f}\nintercept: {c:.2f}")

    print()

    _ = plt.plot(weeks, fvc, 'o', label='Original data', markersize=10)

    _ = plt.plot(weeks, m * weeks + c, 'r', label='Fitted line')

    _ = plt.legend()

    _ = plt.xlabel("Weeks"), plt.ylabel("FVC")

    plt.show()
# gradient = rate of decay in FVC values

sample_best_fit_line_weeks_vs_fvc()
gradients = {} 

agss_vectors = {} 

patients = []



for i, patient_id in enumerate(train.Patient.unique()):

    sub = train.loc[train.Patient == patient_id, :] 

    fvc = sub.FVC.values

    weeks = sub.Weeks.values

    c = np.c_[weeks, np.ones(len(weeks))]

    gradient, intercept = np.linalg.lstsq(c, fvc, rcond=-1)[0]

    

    gradients[patient_id] = gradient

    agss_vectors[patient_id] = get_agss_vector(sub)

    patients.append(patient_id)
def get_img(path):

    d = pydicom.dcmread(path)

    return cv2.resize(d.pixel_array / 2**11, (512, 512))
# sample

_ = plt.imshow(get_img(os.path.join(ROOT, "train", "ID00007637202177411956430", "1.dcm")))
from tensorflow.keras.layers import (

    Input,

    Activation,

    LeakyReLU,

    Dropout,

    BatchNormalization,

    Dense,

    Conv2D, 

    AveragePooling2D,

    GlobalAveragePooling2D,

    Add,

    Flatten,

    Concatenate,

)

import efficientnet.tfkeras as efn



def get_efficientnet(model, shape):

    models_dict = {

        'b0': efn.EfficientNetB0(input_shape=shape, weights=None, include_top=False),

        'b1': efn.EfficientNetB1(input_shape=shape, weights=None, include_top=False),

        'b2': efn.EfficientNetB2(input_shape=shape, weights=None, include_top=False),

        'b3': efn.EfficientNetB3(input_shape=shape, weights=None, include_top=False),

        'b4': efn.EfficientNetB4(input_shape=shape, weights=None, include_top=False),

        'b5': efn.EfficientNetB5(input_shape=shape, weights=None, include_top=False),

        'b6': efn.EfficientNetB6(input_shape=shape, weights=None, include_top=False),

        'b7': efn.EfficientNetB7(input_shape=shape, weights=None, include_top=False),

    }

    return models_dict[model]



def build_model(shape=(512, 512, 1), model_class=None):

    

    img_inp = Input(shape=shape, name="image_input")

    base = get_efficientnet(model_class, shape)

    x = base(img_inp)

    img_outp = GlobalAveragePooling2D()(x)

    

    # AGSS = Age + Gender + SmokingStatus

    agss_inp = Input(shape=(4,), name="age_gender_smokingsstatus_input")

    agss_outp = tf.keras.layers.GaussianNoise(0.2)(agss_inp)

    

    x = Concatenate()([img_outp, agss_outp]) 

    x = Dropout(0.5)(x) 

    output = Dense(1)(x)

    

    model = Model([img_inp, agss_inp] , output)

    weights = [w for w in os.listdir('../input/osic-model-weights') if model_class in w]

    assert len(weights) == 1, "More than one model weights match the 'model_class' substring"

    model.load_weights('../input/osic-model-weights/' + weights[0])

    

    return model



model_classes = ['b5']  # ['b0','b1','b2','b3',b4','b5','b6','b7']

models = [build_model(shape=(512, 512, 1), model_class=m) for m in model_classes]

print('Number of models: ' + str(len(models)))
models[0].summary()
tf.keras.utils.plot_model(

    models[0], 

    to_file='model.png',

    show_shapes=False, 

    show_layer_names=True,

    rankdir='TB',

    expand_nested=False, 

    dpi=120,

)
train_patients, validation_patients = train_test_split(patients, shuffle=True, train_size=0.8)
sns.distplot(list(gradients.values()));
DFs = {

    "train": train,

    "test": test,

}
def fetch_images(patient_id, root=ROOT):

    image_files = os.listdir(os.path.join(root, f'train/{patient_id}/'))

    images = read_images_in_middle_of_scan(image_files, patient_id)

    return images



def read_images_in_middle_of_scan(image_files, patient_id, lower=0.15, upper=0.8):

    images = []

    for filename in image_files:

        file_no, _ = os.path.splitext(filename) # cut out '.dcm' file extension

        file_no = int(file_no)

        is_img_slice_in_middle = lower < file_no / len(image_files) < upper

        if is_img_slice_in_middle:

            image_filepath = os.path.join(ROOT, f'train/{patient_id}/{filename}')

            images.append(get_img(image_filepath))

    return images



def create_agss_vec_mat(patient_df, num_rows):

    agss_vector = get_agss_vector(patient_df)

    agss_matrix = np.array([agss_vector] * num_rows)

    return agss_vector, agss_matrix



def filter_df_with_patient_id(df, patient_id, patient_col="Patient"):

    return df.loc[df[patient_col] == patient_id, :]



def pred_fvc(x, m, c):

    """

    x --> weeks from base week

    m --> gradient i.e. rate of FVC decay (would be -ve for a patient with disease)

    c --> base week FVC

    """

    return m * x + c



def pred_confidence(base_percent, m, gap_in_weeks):

    """

    Predict confidence AKA "std deviation". Lower val means high confidence in predicted FVC.

    base_percent --> percentage in the base week

    m --> gradient i.e. rate of FVC decay (would be -ve for a patient with disease)

    gap_in_weeks --> just the gap irrespective of whether in the past or future

    """

    

    # the formula takes into account that as prediction moves away from the base week,

    # confidence drops (value gets bigger since m is or would be for most -ve)

    return base_percent - m * abs(gap_in_weeks)



def score(fvc_true, fvc_pred, sigma):

    sigma_clip = np.maximum(sigma, 70) # changed from 70, trie 66.7 too

    delta = np.abs(fvc_true - fvc_pred)

    delta = np.minimum(delta, 1000)

    sq2 = np.sqrt(2)

    metric = (delta / sigma_clip) * sq2 + np.log(sigma_clip * sq2)

    return np.mean(metric)



@lru_cache(1000)

def make_model_pred(df_name, patient_id, model_idx):

    global DFs

    df = DFs[df_name]

    patient_df = df[df.Patient == patient_id]

    images = fetch_images(patient_id)

    images = np.expand_dims(images, axis=-1)

    agss_vector, agss_matrix = create_agss_vec_mat(patient_df, num_rows=images.shape[0])

    return models[model_idx].predict([images, agss_matrix])
def calc_patient_score(df_name, patient_id, quantile, model_idx, return_extra_vals=False):

    global DFs

    df = DFs[df_name]

    patient_df = df[df.Patient == patient_id]

    assert not patient_df.empty

    

    # model predicts for each image + agss_vector input

    gradients = make_model_pred(df_name, patient_id, model_idx)

    

    if gradients is None:

        return  # if no valid images in range, it will be None

    gradient = np.quantile(gradients, quantile)  # gradient @ quantile from gradients



    percent_true = patient_df.Percent.values

    fvc_true = patient_df.FVC.values

    weeks_true = patient_df.Weeks.values

    base_week = base_weeks_test[patient_id]



    predicted_fvc = pred_fvc(x=(weeks_true - weeks_true[0]), 

                             m=gradient, 

                             c=fvc_true[0],

                            )

    predicted_confidence = pred_confidence(base_percent=percent_true[0], 

                                 m=gradient, 

                                 gap_in_weeks=(weeks_true - weeks_true[0]),

                                )

    patient_score = score(fvc_true, predicted_fvc, predicted_confidence)

    if not return_extra_vals:

        return patient_score

    else:

        return patient_score, gradient, fvc_predict, confidence
subs = []

start = datetime.datetime.now()

for model_idx in range(len(models)):

    quantile_means = []

    quantiles = np.arange(0.1, 1.0, 0.05)

    for quantile in quantiles:

        

        print(f"Quantile: {quantile:.2f}", end=" -->  ")

        patient_scores_per_quantile = []

        

        for patient_id in validation_patients:

            if patient_id in ['ID00011637202177653955184', 'ID00052637202186188008618']:

                continue

            one_patient_score_per_quantile = calc_patient_score("train", 

                                                                patient_id, 

                                                                quantile, model_idx,

                                                               )

            if one_patient_score_per_quantile is not None:

                patient_scores_per_quantile.append(one_patient_score_per_quantile)



        mean_quantile_score = np.mean(patient_scores_per_quantile)

        print(f"Patient scores mean for quantile {quantile:.2f}: {mean_quantile_score:.4f}")

        quantile_means.append(mean_quantile_score)



    sub = pd.read_csv(os.path.join(ROOT,'sample_submission.csv'))

    test = pd.read_csv(os.path.join(ROOT,'test.csv'))



    ## quantile with the smallest mean -> smallest error

    lowest_quantile_mean_idx = np.argmin(quantile_means)

    lowest_quantile = (lowest_quantile_mean_idx + 1) / 10



    gradient_test, calc_fvc_base_test, percent_test, base_weeks_test = {}, {}, {}, {}

    

    # this loop defines base parameters for each patient needed to calculate week-by-week prediction

    for patient_id in test.Patient.unique():

        _, gradient, *_ = calc_patient_score("test", 

                                             patient_id, 

                                             lowest_quantile,

                                             model_idx,

                                             return_extra_vals=True,

                                            )  # only gradient needed

        patient_df = test[test.Patient == patient_id]

        

        # test assumption: df will have 1 row since test set

        assert patient_df.shape[0] == 1

        

        gradient_test[patient_id] = gradient  # prediction value of the model

        

        # pred of FVC at week 0 itself. Other weeks will be predicted using this as base

        calc_fvc_base_test[patient_id] = (patient_df.FVC.values - 

                                          gradient * patient_df.Weeks).values[0]  



        percent_test[patient_id] = patient_df.Percent.values[0]

        base_weeks_test[patient_id] = patient_df.Weeks.values[0]



    # this loop predicts values (FVC and confidence) for each patient's each week

    for k in sub.Patient_Week.values:

        

        patient_id, week_no = k.split('_')

        week_no = int(week_no)

        

        gradient = gradient_test[patient_id]

        base_fvc = calc_fvc_base_test[patient_id]

        base_percent = percent_test[patient_id]

        base_week = base_weeks_test[patient_id]

        gap_from_base_week = base_week - week_no

        

        predicted_fvc = pred_fvc(week_no, m=gradient, c=base_fvc)

        predicted_conf = pred_confidence(base_percent,

                                         m=gradient,

                                         gap_in_weeks=gap_from_base_week,

                                        )

        

        sub.loc[sub.Patient_Week==k, 'FVC'] = predicted_fvc

        sub.loc[sub.Patient_Week==k, 'Confidence'] = predicted_conf

    

    sub_ = sub[["Patient_Week", "FVC", "Confidence"]].copy()

    subs.append(sub_)

end = datetime.datetime.now()

print(end - start)
N = len(subs)

sub = subs[0].copy() # ref

sub["FVC"] = 0

sub["Confidence"] = 0

for i in range(N):

    sub["FVC"] += subs[0]["FVC"] * (1/N)

    sub["Confidence"] += subs[0]["Confidence"] * (1/N)
sub.head()
img_sub = sub[["Patient_Week","FVC","Confidence"]].copy()

img_sub.to_csv("submission_img.csv", index=False)
BATCH_SIZE = 128



tr = pd.read_csv(f"{ROOT}/train.csv")

tr.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])

chunk = pd.read_csv(f"{ROOT}/test.csv")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")



sub[['Patient', 'Weeks']] = sub['Patient_Week'].str.split("_", expand=True)

sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]

sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")



tr['WHERE'] = 'train'

chunk['WHERE'] = 'val'

sub['WHERE'] = 'test'

data = tr.append([chunk, sub])



names = ["train", "val", "test", "combined"]

for i, df in enumerate([tr, chunk, sub, data]):

    df_shape_in_blue = (Fore.BLUE, df.shape, Style.RESET_ALL)

    uniq_p_in_green = (Fore.GREEN, df.Patient.nunique(), Style.RESET_ALL)

    print(names[i], "-> shape", *df_shape_in_blue, " -> unique patients", *uniq_p_in_green)
# add minimum week for all patients. The actual one. Submission (called test here)

# contains all possible weeks. But that is just necessary for predictions.

# The actual one is the no. of weeks before/after the CT-Scan, the patient went for FVC measurement

data['min_week'] = data['Weeks']

data.loc[data.WHERE=='test', 'min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')



base = data.loc[data.Weeks == data.min_week, ['Patient','FVC']].drop_duplicates()

data = data.merge(base, how="left", on="Patient")

data.rename({"FVC_x": "FVC", "FVC_y": "min_FVC"}, axis=1, inplace=True)

data['from_base_week'] = data['Weeks'].astype(int) - data['min_week']

del base

data.head()
data = pd.get_dummies(data, columns=["Sex", "SmokingStatus"], prefix="", prefix_sep="", )

data.head()
def min_max_scaler(df, col):

    scaler = MinMaxScaler()

    col_matrix = np.expand_dims(df.loc[:, col].values, axis=-1)  # sklearn-requirement

    scaler.fit(col_matrix)

    return scaler.transform(col_matrix).squeeze()
data['age'] = min_max_scaler(data, "Age")

data['BASE'] = min_max_scaler(data, "min_FVC")

data['week'] = min_max_scaler(data, "from_base_week")

data['percent'] = min_max_scaler(data, "Percent")

data.head()
tr = data.loc[data.WHERE=='train']

chunk = data.loc[data.WHERE=='val']

sub = data.loc[data.WHERE=='test']

del data



tr.shape, chunk.shape, sub.shape
SIGMA_LOWER_LIMIT = tf.constant(70, dtype='float32')

MAX_ABS_ERROR = tf.constant(1000, dtype="float32")

QUANTILES = tf.constant(np.array([[0.2, 0.5, 0.8]]), dtype=tf.float32)



def score(y_true, y_pred):

    tf.dtypes.cast(y_true, tf.float32)

    tf.dtypes.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    sigma_clip = tf.maximum(sigma, SIGMA_LOWER_LIMIT)

    delta = tf.abs(y_true[:, 0] - fvc_pred)

    delta = tf.minimum(delta, MAX_ABS_ERROR)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32))

    

    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)

    return K.mean(metric)



# The pinball loss function is a metric used to assess the accuracy of a quantile forecast. 



def quantile_loss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    e = y_true - y_pred

    v = tf.maximum(QUANTILES * e, (QUANTILES - 1) * e)

    return K.mean(v)



def mloss(_lambda):

    def loss(y_true, y_pred):

        return _lambda * quantile_loss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss



def make_model(nh):

    z = L.Input((nh,), name="Patient")

    x = L.Dense(100, activation="relu", name="d1")(z)

    x = L.Dense(100, activation="relu", name="d2")(x)

    p1 = L.Dense(3, activation="linear", name="p1")(x)  # 2 different activations

    p2 = L.Dense(3, activation="relu", name="p2")(x)

    

    # lambda layer takes in this case one input x (a list of outputs [p1, p2])

    # keep output of linear activation (p1 i.e. x[0]) 

    # add cumsum of relu activation (p2) to p1 --> axis=1 means add horizontally (values of same sample)

    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

    

    model = M.Model(z, preds, name="CNN")

    model.compile(loss=mloss(0.8), 

                  optimizer=tf.keras.optimizers.Adam(lr=0.1, 

                                                     beta_1=0.9,

                                                     beta_2=0.999, 

                                                     epsilon=None, 

                                                     decay=0.01, 

                                                     amsgrad=False,

                                                    ), 

                  metrics=[score],

                 )

    return model
FEATURE_COLS = tr.columns[tr.columns.get_loc("from_base_week") + 1:].tolist()

FEATURE_COLS
y = tr['FVC'].astype(np.float32).values

X = tr[FEATURE_COLS].values

test = sub[FEATURE_COLS].values

num_features = X.shape[1]

pred_test = np.zeros((test.shape[0], 3))

pred_val = np.zeros((X.shape[0], 3))
net = make_model(num_features)

net.summary()  # each input datapoint will have 3 output values (3 quantiles)
tf.keras.utils.plot_model(

    net, 

    to_file='model.png',

    show_shapes=False, 

    show_layer_names=True,

    rankdir='TB',

    expand_nested=False, 

    dpi=120,

)
NFOLD = 5

kf = KFold(n_splits=NFOLD)
%%time



EPOCHS = 800



for fold_no, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):

    

    print(f"FOLD {fold_no}")

    

    net = make_model(num_features)

    

    X_train, y_train = X[tr_idx], y[tr_idx]

    X_val, y_val = X[val_idx], y[val_idx]

    

    net.fit(x=X_train, 

            y=y_train, 

            batch_size=BATCH_SIZE, 

            epochs=EPOCHS, 

            validation_data=(X_val, y_val),

            verbose=0,

           )

    

    print("train", net.evaluate(X_train, y_train, verbose=0, batch_size=BATCH_SIZE))

    print("val", net.evaluate(X_val, y_val, verbose=0, batch_size=BATCH_SIZE))

    

    print("predict val...")

    pred_val[val_idx] = net.predict(X_val, batch_size=BATCH_SIZE, verbose=0)

    

    print("predict test...")

    fold_prediction = net.predict(test, batch_size=BATCH_SIZE, verbose=0)

    fold_prediction_normalized = fold_prediction / NFOLD

    pred_test += fold_prediction_normalized
# prediction for each data point consists of 3 values (i.e. 3 quartiles)



sigma_opt = mean_absolute_error(y, pred_val[:, 1])

unc = pred_val[:, 2] - pred_val[:, 0]

sigma_mean = np.mean(unc)

print(sigma_opt, sigma_mean)



idxs = np.random.randint(0, y.shape[0], 100)

plt.figure(figsize=(10, 8))

plt.plot(y[idxs], label="ground truth")

plt.plot(pred_val[idxs, 0], label="q25")

plt.plot(pred_val[idxs, 1], label="q50")

plt.plot(pred_val[idxs, 2], label="q75")

plt.legend(loc="best")

plt.show()
print(unc.min(), unc.mean(), unc.max(), (unc>=0).mean())
plt.hist(unc)

plt.title("uncertainty in prediction")

plt.show()
sub.head()
# PREDICTION

sub['FVC1'] = 1. * pred_test[:, 1]

sub['Confidence1'] = pred_test[:, 2] - pred_test[:, 0]

subm = sub[['Patient_Week', 'FVC', 'Confidence', 'FVC1', 'Confidence1']].copy()

assert subm.FVC1.isna().sum() == 0

subm.head(10)
subm.loc[:, 'FVC'] = subm.loc[:, 'FVC1']

if sigma_mean < 70:

    subm['Confidence'] = sigma_opt

else:

    subm.loc[:, 'Confidence'] = subm.loc[:,'Confidence1']
subm.head()
subm.describe().T
otest = pd.read_csv(os.path.join(ROOT, 'test.csv'))



for i in range(len(otest)):

    

    patient_week = otest.Patient[i] + '_' + str(otest.Weeks[i])

    is_patient_week_row = subm['Patient_Week'] == patient_week

    

    subm.loc[is_patient_week_row, 'FVC'] = otest.FVC[i]

    subm.loc[is_patient_week_row, 'Confidence'] = 0.1
reg_sub = subm[["Patient_Week","FVC","Confidence"]].copy()

reg_sub.to_csv("submission_regression.csv", index=False)
df1 = img_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)

df2 = reg_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
df = df1[['Patient_Week']].copy()

df['FVC'] = 0.2666 * df1['FVC'] + 0.2444 * df2['FVC']

df['Confidence'] = 0.2666 * df1['Confidence'] + 0.7444 * df2['Confidence']

df.head()
df.to_csv('submission.csv', index=False)