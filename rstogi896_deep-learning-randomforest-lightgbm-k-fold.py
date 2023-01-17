import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn import metrics

from sklearn.metrics import roc_auc_score

from sklearn.utils import shuffle

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

import matplotlib

from sklearn.metrics import mean_squared_error

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

import os

import time

import numpy as np

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

import lightgbm as lgb

sns.set()

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import warnings

warnings.filterwarnings("ignore")
train=pd.read_csv("../input/X_train.csv")

train.head()
test=pd.read_csv("../input/X_test.csv")

test.head()
sample_submission=pd.read_csv("../input/sample_submission.csv")

sample_submission.head()
test_id=sample_submission['series_id']

y_train=pd.read_csv("../input/y_train.csv")

y_train.head()
y_train['surface'].value_counts().plot.bar();
train.shape,test.shape,y_train.shape
plt.figure(figsize=(26, 20))

for i, col in enumerate(train.columns[3:]):

    plt.subplot(3, 4, i + 1)

    plt.plot(train.loc[train['series_id'] == 1, col])

    plt.title(col)
plt.figure(figsize=(26, 20))

for i, col in enumerate(test.columns[3:]):

    plt.subplot(3, 4, i + 1)

    plt.plot(test.loc[test['series_id'] == 1, col])

    plt.title(col)
# refrence from https://www.kaggle.com/jsaguiar/surface-recognition-baseline

def feature_extraction(raw_frame):

    frame = pd.DataFrame()

    raw_frame['angular_velocity'] = raw_frame['angular_velocity_X'] + raw_frame['angular_velocity_Y'] + raw_frame['angular_velocity_Z']

    raw_frame['linear_acceleration'] = raw_frame['linear_acceleration_X'] + raw_frame['linear_acceleration_Y'] + raw_frame['linear_acceleration_Y']

    raw_frame['velocity_to_acceleration'] = raw_frame['angular_velocity'] / raw_frame['linear_acceleration']

    

    for col in raw_frame.columns[3:]:

        frame[col + '_mean'] = raw_frame.groupby(['series_id'])[col].mean()

        frame[col + '_std'] = raw_frame.groupby(['series_id'])[col].std()

        frame[col + '_max'] = raw_frame.groupby(['series_id'])[col].max()

        frame[col + '_min'] = raw_frame.groupby(['series_id'])[col].min()

        frame[col + '_max_to_min'] = frame[col + '_max'] / frame[col + '_min']

        

        frame[col + '_mean_abs_change'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        frame[col + '_abs_max'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

    return frame
train_df = feature_extraction(train)

test_df = feature_extraction(test)

train_df.head()
train_df.shape,test_df.shape
train_df["orientation_X_mean"].hist()
train_df["orientation_X_std"].hist()
test_df["orientation_X_mean"].hist()
test_df["orientation_X_std"].hist()
le = LabelEncoder()

target_train = le.fit_transform(y_train['surface'])
train_df['surface']=target_train

sns.violinplot(data=train_df,x="surface", y="orientation_X_mean")
sns.violinplot(data=train_df,x="surface", y="orientation_Y_mean")
sns.violinplot(data=train_df,x="surface", y="orientation_Z_mean")
sns.violinplot(data=train_df,x="surface", y="angular_velocity_X_mean")
sns.violinplot(data=train_df,x="surface", y="angular_velocity_Z_mean")
train_df=train_df.drop(['surface'],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(train_df, target_train, random_state = 0)

X_train.shape,X_test.shape
sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)
preds = []

K = 12

kf = KFold(n_splits = K, random_state = 3228, shuffle = True)
alg =  RandomForestClassifier()
for train_index, test_index in kf.split(X_train):

    train_X, valid_X = X_train[train_index], X_train[test_index]

    train_y, valid_y = Y_train[train_index], Y_train[test_index]

    alg.fit( train_X,  train_y)                   

    pred = alg.predict(X_test)

    preds.append(list(pred))
predx=[]

for i in range(len(preds[0])):

    sum=[]

    for j in range(K):

        sum.append(preds[j][i])

            

    predx.append(max(set(sum), key =sum.count))
accuracy_score(Y_test, predx)
cm = confusion_matrix(Y_test, predx)

cm 
params = {

    'num_leaves': 20,

    'min_data_in_leaf': 15,

    'objective': 'multiclass',

    'max_depth': 8,

    'learning_rate': 0.01,

    "boosting": "gbdt",

    "bagging_freq": 5,

    "bagging_fraction": 0.8126672064208567,

    "bagging_seed": 11,

    "verbosity": -1,

    'reg_alpha': 0.1302650970728192,

    'reg_lambda': 0.3603427518866501,

    "num_class": 9,

    'nthread': -1

}



def multiclass_accuracy(preds, train_data):

    labels = train_data.get_label()

    pred_class = np.argmax(preds.reshape(9, -1).T, axis=1)

    return 'multi_accuracy', np.mean(labels == pred_class), True



t0 = time.time()

train_set = lgb.Dataset(X_train, label=Y_train)

eval_hist = lgb.cv(params, train_set, nfold=8, num_boost_round=1200,

                   early_stopping_rounds=80, seed=19, feval=multiclass_accuracy)

num_rounds = len(eval_hist['multi_logloss-mean'])

# retrain the model and make predictions for test set

clf = lgb.train(params, train_set, num_boost_round=num_rounds)

predictions = clf.predict(X_test, num_iteration=None)

print("Timer: {:.1f}s".format(time.time() - t0))
idx = predictions.argmax(axis=1)

y_pred1 = (idx[:,None] == np.arange(predictions.shape[1])).astype(int)

y_pred1
y_pred1 = [np.where(r == 1)[0][0] for r in y_pred1]
accuracy_score(Y_test, y_pred1)
cm = confusion_matrix(Y_test, y_pred1)

cm 
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.optimizers import SGD
Y_traind = np.zeros((2857, 9))

Y_traind[np.arange(2857), Y_train] = 1

Y_traind
predsd = []

K = 3

kf = KFold(n_splits = K, random_state = 3228, shuffle = True)
for train_index, test_index in kf.split(X_train):

    train_X, valid_X = X_train[train_index], X_train[test_index]

    train_y, valid_y = Y_train[train_index], Y_train[test_index]

    classifier = Sequential()

    classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu', input_dim = 91))

    classifier.add(Dropout(0.1))

    classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))

    classifier.add(Dropout(0.1))

    classifier.add(Dense(output_dim = 40, init = 'uniform', activation = 'relu'))

    classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'softmax'))

    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    classifier.fit(X_train,Y_traind, nb_epoch = 130)

    pred = classifier.predict(X_test)

    predsd.append(list(pred))
f=0

for i in range(K):

    a=predsd[i]

    f+=np.array(a)

f=f/K; 

f=np.array(f)
idx = f.argmax(axis=1)

y_predx = (idx[:,None] == np.arange(f.shape[1])).astype(int)

y_predx
y_predx = [np.where(r == 1)[0][0] for r in y_predx]
accuracy_score(Y_test, y_predx)
cm = confusion_matrix(Y_test, y_predx)

cm
lt=[y_predx,y_pred1,predx]
predx=[]

for i in range(len(lt[0])):

    sum=[]

    for j in range(3):

        

        sum.append(lt[j][i])

            

    predx.append(max(set(sum), key =sum.count))
accuracy_score(Y_test, predx)
params = {

    'num_leaves': 20,

    'min_data_in_leaf': 15,

    'objective': 'multiclass',

    'max_depth': 10,

    'learning_rate': 0.01,

    "boosting": "gbdt",

    "bagging_freq": 5,

    "bagging_fraction": 0.8126672064208567,

    "bagging_seed": 11,

    "verbosity": -1,

    'reg_alpha': 0.1302650970728192,

    'reg_lambda': 0.3603427518866501,

    "num_class": 9,

    'nthread': -1

}



def multiclass_accuracy(preds, train_data):

    labels = train_data.get_label()

    pred_class = np.argmax(preds.reshape(9, -1).T, axis=1)

    return 'multi_accuracy', np.mean(labels == pred_class), True



t0 = time.time()

train_set = lgb.Dataset(train_df, label=target_train)

eval_hist = lgb.cv(params, train_set, nfold=20, num_boost_round=1400,

                   early_stopping_rounds=80, seed=19, feval=multiclass_accuracy)

num_rounds = len(eval_hist['multi_logloss-mean'])

# retrain the model and make predictions for test set

clf = lgb.train(params, train_set, num_boost_round=num_rounds)

predictions = clf.predict(test_df, num_iteration=None)

print("Timer: {:.1f}s".format(time.time() - t0))
predictions
v1, v2 = eval_hist['multi_logloss-mean'][-1], eval_hist['multi_accuracy-mean'][-1]

print("Validation logloss: {:.4f}, accuracy: {:.4f}".format(v1, v2))

plt.figure(figsize=(10, 4))

plt.title("CV multiclass logloss")

num_rounds = len(eval_hist['multi_logloss-mean'])

ax = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_logloss-mean'])

ax2 = ax.twinx()

p = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_logloss-stdv'], ax=ax2, color='r')



plt.figure(figsize=(10, 4))

plt.title("CV multiclass accuracy")

num_rounds = len(eval_hist['multi_accuracy-mean'])

ax = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_accuracy-mean'])

ax2 = ax.twinx()

p = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_accuracy-stdv'], ax=ax2, color='r')
importance = pd.DataFrame({'gain': clf.feature_importance(importance_type='gain'),

                           'feature': clf.feature_name()})

importance.sort_values(by='gain', ascending=False, inplace=True)

plt.figure(figsize=(10, 20))

ax = sns.barplot(x='gain', y='feature', data=importance)
sample_submission['surface'] = le.inverse_transform(predictions.argmax(axis=1))

sample_submission.to_csv('Lgb.csv', index=False)