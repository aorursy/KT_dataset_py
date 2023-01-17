# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

from joblib import Parallel, delayed

from PIL import Image



from sklearn import svm, tree, neighbors, ensemble

from sklearn.cluster import KMeans

from sklearn import model_selection, feature_selection 

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

from sklearn.metrics import auc, accuracy_score, precision_recall_curve, f1_score

from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.semi_supervised import LabelPropagation



import scipy.stats



import lightgbm as lgb

import xgboost as xgb

import catboost as ctb

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.applications.densenet import preprocess_input, DenseNet121

from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D

import keras.backend as K



IMG_SIZE = 256



model_input = Input((IMG_SIZE,IMG_SIZE,3))

backbone = DenseNet121(input_tensor = model_input, 

                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",

                       include_top = False)

x = backbone.output

x = GlobalAveragePooling2D()(x)

x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)

x = AveragePooling1D(4)(x)

model_output = Lambda(lambda x: x[:,:,0])(x)



m = Model(model_input,model_output)
def load_image(path):

    img = Image.open(path)

    img = img.resize((IMG_SIZE, IMG_SIZE))

    #preprocessing

    return np.asarray(img)



batch_size = 32

labels = ['cgm', 'cmd', 'healthy', 'cbb', 'cbsd']



#train

train  = pd.DataFrame(columns=['target'] + ['DenseNet_%d'%(i) for i in range(256)], dtype='float32')

for lb in range(len(labels)):

    image_files = glob.glob('../input/cassava-disease/train/train/' + labels[lb] + '/*.jpg')

    n_batches = len(image_files) // batch_size + 1

    

    for i in tqdm(range(n_batches)):

        batch = image_files[i*batch_size:(i+1)*batch_size]

        batch_images = np.zeros((len(batch), IMG_SIZE, IMG_SIZE, 3))

        for i,p in enumerate(batch):

            batch_images[i] = load_image(p)

        preds = m.predict(batch_images)

        for i,p in enumerate(batch):

            image_id = p.split('/')[-1]

            train.loc[image_id, 'target'] = lb

            train.loc[image_id, 1:] = preds[i]

            

#test

submission = pd.read_csv('../input/cassava-disease/sample_submission_file.csv')

test = pd.DataFrame(index=submission['Id'], columns=['DenseNet_%d'%(i) for i in range(256)])

image_files = glob.glob('../input/cassava-disease/test/test/0/*.jpg')

n_batches = len(image_files) // batch_size + 1



for i in tqdm(range(n_batches)):

    batch = image_files[i*batch_size:(i+1)*batch_size]

    batch_images = np.zeros((len(batch), IMG_SIZE, IMG_SIZE, 3))

    for i,p in enumerate(batch):

        batch_images[i] = load_image(p)

    preds = m.predict(batch_images)

    for i,p in enumerate(batch):

        image_id = p.split('/')[-1]

        test.loc[image_id, :] = preds[i]



#extra

extra = pd.DataFrame(columns=['DenseNet_%d'%(i) for i in range(256)])

image_files = glob.glob('../input/cassava-disease/extraimages/extraimages/*.jpg')

n_batches = len(image_files) // batch_size + 1



for i in tqdm(range(n_batches)):

    batch = image_files[i*batch_size:(i+1)*batch_size]

    batch_images = np.zeros((len(batch), IMG_SIZE, IMG_SIZE, 3))

    for i,p in enumerate(batch):

        batch_images[i] = load_image(p)

    preds = m.predict(batch_images)

    for i,p in enumerate(batch):

        image_id = p.split('/')[-1]

        extra.loc[image_id, :] = preds[i]



print("train:", train.shape)

print("test:", test.shape)

print("extra:", extra.shape)
def run_lgb(params, X_train, X_test, extra=None, n_splits=5):

    verbose_eval = 1000

    num_rounds = 200000

    early_stop = 1000

    

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)

    

    oof_train = np.zeros((X_train.shape[0]))

    oof_test  = np.zeros((X_test.shape[0], n_splits))

    

    i = 0

    for train_idx, valid_idx in kf.split(X_train, X_train['target']):

        X_tr = X_train.iloc[train_idx].drop('target', axis=1)

        X_val = X_train.iloc[valid_idx].drop('target', axis=1)

        

        y_tr = X_train.iloc[train_idx]['target']

        y_val = X_train.iloc[valid_idx]['target']

        

        if extra is not None:

            X_tr = pd.concat([X_tr, extra.drop('target', axis=1)], sort=False)

            y_tr = pd.concat([y_tr, extra['target']], sort=False)

        

        d_train = lgb.Dataset(data=X_tr, label=y_tr)

        d_valid = lgb.Dataset(data=X_val, label=y_val)

        

        watchlist = [d_train, d_valid]

        model = lgb.train(params,

                          train_set=d_train,

                          num_boost_round=num_rounds,

                          valid_sets=watchlist,

                          early_stopping_rounds=early_stop, verbose_eval=verbose_eval)

        valid_pred = model.predict(X_val.values, num_iteration=model.best_iteration)

        test_pred = model.predict(X_test.values, num_iteration=model.best_iteration)

        

        oof_train[valid_idx] = np.argmax(valid_pred, axis=1)

        oof_test[:,i] = np.argmax(test_pred, axis=1)

        i += 1

    return model, oof_train, oof_test
lgb_params = {

    "objective" : "multiclass",

    "num_class" : len(labels),

    "metric" : "multi_logloss",

    "boosting": 'gbdt',

    "max_depth" : -1,

    "num_leaves" : 30,

    "learning_rate" : 0.01,

    "bagging_freq": 5,

    "bagging_fraction" : 0.85,

    "feature_fraction" : 0.4,

#    "min_data_in_leaf": 20,

#    "min_sum_heassian_in_leaf": 10,

    "tree_learner": "serial",

    "boost_from_average": "false",

    "bagging_seed" : 10,

    "verbosity" : 1,

    'seed': 1337,

#    'max_bin':100,

    'lambda_l1': 2.622427756417558,

    'lambda_l2': 2.624427931714477,

}



model_lgb, oof_train_lgb, oof_test_lgb = run_lgb(lgb_params, train, test)
print("CV Score:", accuracy_score(train['target'], oof_train_lgb))
submission = pd.read_csv('../input/cassava-disease/sample_submission_file.csv')

submission['Category'] = scipy.stats.mode(oof_test_lgb, axis=1)[0].astype('int').flatten()

submission['Category'] = submission['Category'].apply(lambda x: labels[x])

submission.head(5)
submission.to_csv('submission_lgb.csv', index=False)
tmp = pd.concat([train, extra], sort=False)

tmp = tmp.fillna(-1)
label_prop = LabelPropagation(kernel='knn', n_neighbors=2)

#label_prop = LabelPropagation(n_neighbors=2)

label_prop.fit(tmp.drop('target', axis=1), tmp['target'])



preds = label_prop.predict(tmp.drop('target', axis=1))

pd.value_counts(preds)
extra_ = extra.copy().astype('float32')

extra_['target'] = preds[tmp['target']==-1]
model_ex, oof_train_ex, oof_test_ex = run_lgb(lgb_params, train, test, extra=extra_)
print("CV Score:", accuracy_score(train['target'], oof_train_ex))
submission = pd.read_csv('../input/cassava-disease/sample_submission_file.csv')

submission['Category'] = scipy.stats.mode(oof_test_ex, axis=1)[0].astype('int').flatten()

submission['Category'] = submission['Category'].apply(lambda x: labels[x])

submission.head(5)
submission.to_csv('submission_lp.csv', index=False)