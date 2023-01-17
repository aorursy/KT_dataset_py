# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline, FeatureUnion

import category_encoders as ce

from sklearn.base import TransformerMixin, BaseEstimator

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import catboost as cb



import lightgbm as lgb

from tensorflow.keras.utils import to_categorical
# input

demo = pd.read_csv('../input/tj19data/demo.csv')

train = pd.read_csv('../input/tj19data/train.csv')

test = pd.read_csv('../input/tj19data/test.csv')

txn = pd.read_csv('../input/tj19data/txn.csv')



# output

submit = pd.read_csv('../input/tj19data/sample_submission_v1.csv')
train.head()
test.head()
demo.head()
txn.head()
# output

submit.head()
train['label'].value_counts()
pd.DataFrame({'nunique':demo.nunique(), '% missing':demo.isnull().mean()})
# [c3, c4] -> [n1, n2]

demo.sample(10)
# n0 -> age

demo.n0.hist()

plt.show()
def get_demo_dummy():

    demo = pd.read_csv('../input/tj19data/demo.csv')



    demo['c0'] -= 1

    demo = pd.get_dummies(demo, columns=['c1', 'c2'])

    demo.fillna(0, inplace=True)

    demo['max_n1_n2'] = demo[['n1', 'n2']].max(axis=1)



    demo.drop(['n1', 'n2'], axis=1, inplace=True)

    

    return demo



def get_train_test():

    demo = get_demo_dummy()

    train = pd.read_csv('../input/tj19data/train.csv')

    test = pd.read_csv('../input/tj19data/test.csv')

    train = pd.merge(train, demo, on='id', how='left')

    test = pd.merge(test, demo, on='id', how='left')

    return train.drop('id', axis=1), test.drop('id', axis=1)





train, test = get_train_test()
y = train.pop('label')

clf = lgb.LGBMClassifier().fit(train, y)
fig, ax = plt.subplots(figsize=(12, 8))

pd.Series(clf.feature_importances_, index=train.columns).sort_values(ascending=False).plot.barh(ax=ax)
txn.query('c5==12').sample(10)
txn.c6.value_counts(normalize=True)
# [old_cc_no, old_cc_label, n3]

txn[['old_cc_no', 'n7']].nunique()
def day2month(x):

    days = np.array([31,29,31,30,30,30,31,31,30,31,30,31]).cumsum()

    month = np.arange(1, 13)



    for n, d in zip(month, days):

        if x<=d:

            return n

        

def get_txn():

    txn = pd.read_csv('../input/tj19data/txn.csv')

    

    txn_prep = txn.groupby('id').old_cc_no.nunique()

    txn['day_of_week'] = txn['n3'] % 7

    txn['month'] = txn['day_of_week'].map(day2month)

    for c in ['n4', 'n5', 'n6']:

        # n4_by_id

        n4_by_id = txn.groupby('id')[c].sum()

        

        # n4 by id+c5

        n4_by_id_c5 = txn.groupby(['id', 'c5'])[c].sum()

        n4_by_id_c5 = n4_by_id_c5.unstack().fillna(0)

        n4_by_id_c5.columns = n4_by_id_c5.columns.name + '_' + n4_by_id_c5.columns.astype(str) +'_sum'

        temp = (n4_by_id_c5 > 0).mean(axis=0)

        n4_by_id_c5 = n4_by_id_c5[temp[temp > 0.05].index]



        # n4 by id+c6

        #n4_by_id_c6 = txn.groupby(['id', 'c6'])[c].sum()

        #n4_by_id_c6 = n4_by_id_c6.unstack().fillna(0)

        #n4_by_id_c6.columns = n4_by_id_c6.columns.name + '_' + n4_by_id_c6.columns.astype(str) +'_sum'

        #temp = (n4_by_id_c6 > 0).mean(axis=0)

        #n4_by_id_c6 = n4_by_id_c6[temp[temp > .05].index] 



        # n4 by id+c7

        #n4_by_id_c7 = txn.groupby(['id', 'c7'])[c].sum().unstack().fillna(0)

        #n4_by_id_c7.columns = n4_by_id_c7.columns.name + '_' + n4_by_id_c7.columns.astype(str) +'_sum'

        #temp = (n4_by_id_c7 > 0).mean(axis=0)

        #n4_by_id_c7 = n4_by_id_c7[temp[temp > .05].index] 



        # n4 by id+dow

        n4_by_id_dayofweek = txn.groupby(['id', 'day_of_week'])[c].mean().unstack().fillna(0)

        n4_by_id_dayofweek.columns = n4_by_id_dayofweek.columns.name + '_' + n4_by_id_dayofweek.columns.astype(str) + '_mean'



        # n4 by id+month

        n4_by_id_month = txn.groupby(['id', 'day_of_week'])[c].sum().unstack().fillna(0)

        n4_by_id_month.columns = n4_by_id_month.columns.name + '_' + n4_by_id_month.columns.astype(str) + '_sum'



        temp = pd.concat([n4_by_id, n4_by_id_c5, 

                          #n4_by_id_c6, 

                          #n4_by_id_c7,

                          n4_by_id_dayofweek, n4_by_id_month], axis=1)

        txn_prep = pd.merge(txn_prep, temp, left_index=True, right_index=True)

    for c in ['c5', 'c6', 'c7']:

        c_by_id = txn.groupby('id')[c].value_counts(normalize=True).unstack().fillna(0)



        c_by_id.columns = c_by_id.columns.name + '_' + c_by_id.columns.astype(str) + '_%count'



        temp = (c_by_id > 0).mean(axis=0)



        c_by_id = c_by_id[temp[temp > .05].index]



        txn_prep = pd.merge(txn_prep, c_by_id, left_index=True, right_index=True)

    return txn_prep



def get_train_test_txn():

    txn = get_txn()

    train = pd.read_csv('../input/tj19data/train.csv')

    test = pd.read_csv('../input/tj19data/test.csv')

    train = pd.merge(train, txn, left_on='id', right_index=True, how='left')

    test = pd.merge(test, txn, left_on='id', right_index=True, how='left')

    return train.drop('id', axis=1), test.drop('id', axis=1)





def get_train_test():

    demo = get_demo_dummy()

    txn = get_txn()

    train = pd.read_csv('../input/tj19data/train.csv')

    test = pd.read_csv('../input/tj19data/test.csv')

    

    train = pd.merge(train, txn, left_on='id', right_index=True, how='left')

    test = pd.merge(test, txn, left_on='id', right_index=True, how='left')

    train = pd.merge(train, demo, on='id', how='left')

    test = pd.merge(test, demo, on='id', how='left')

    return train.drop('id', axis=1), test.drop('id', axis=1)
train, test = get_train_test_txn()

y = train.pop('label')

clf = lgb.LGBMClassifier().fit(train, y)
fig, ax = plt.subplots(figsize=(12, 8))

pd.Series(clf.feature_importances_, index=train.columns).sort_values(ascending=False).head(20).plot.barh(ax=ax)
from tensorflow.keras import layers, models, callbacks

import tensorflow.keras.backend as K

from tensorflow.keras.utils import to_categorical
train, test = get_train_test()
y = train.pop('label')



y_cat = to_categorical(y, num_classes=13)



X_train, X_val, y_train_cat, y_val_cat = train_test_split(train, y_cat, stratify=y, random_state=123)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_val = sc.transform(X_val)



X_train.shape, X_val.shape
X_test = sc.transform(test)
def build_model():

    K.clear_session()



    nn = models.Sequential()

    nn.add(layers.Dense(64, activation='relu', input_dim=X_train.shape[1]))

    nn.add(layers.Dropout(0.5))

    nn.add(layers.Dense(32, activation='relu'))

    nn.add(layers.Dropout(0.5))

    nn.add(layers.Dense(32, activation='relu'))

    nn.add(layers.Dropout(0.5))

    nn.add(layers.Dense(13, activation='softmax'))



    nn.compile(loss='categorical_crossentropy')

    

    return nn
nn = build_model()

nn_callbacks = [callbacks.EarlyStopping(patience=15),

                    callbacks.ReduceLROnPlateau(factor=.5, patience=5),

                    callbacks.ModelCheckpoint('current_best_nn.h5', save_best_only=True)]



hx = nn.fit(X_train , y_train_cat, validation_data=(X_val , y_val_cat), 

            #class_weight=w, 

            callbacks=nn_callbacks, 

            epochs=200, batch_size=1024)



nn = models.load_model('current_best_nn.h5')
pd.DataFrame(hx.history).iloc[:, :-1].plot()