# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_path = '../input/cat-in-the-dat/train.csv'

test_path = '../input/cat-in-the-dat/test.csv'
train = pd.read_csv(train_path)

test = pd.read_csv(test_path)

train
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(16,5))

sns.countplot(x = train['target'], data=train)

# 애초에 '0'보다 '1'이 훨씬 적음(절반도 안됨)
sns.countplot(x=train['bin_3'],)
def find_missing_cols(dataframe, list):

    list = []

    columns = dataframe.columns

    for col in columns:

        missing_judgement = dataframe[col].isnull().any()

        if missing_judgement == True:

            list.append(col)

        else:

            pass

    return list



missing_col=[]

missing_col = find_missing_cols(train, missing_col)

missing_col
def find_category_col(dataframe):

    s = (dataframe.dtypes == 'object')

    object_cols = list(s[s].index)

    return object_cols



cat_cols = find_category_col(train)

cat_cols
value_counting = []

for cols in cat_cols:

    a = len(train[cols].unique())

    value_counting.append(a)    

value_counting



values = pd.DataFrame({'columns' : cat_cols, 'values count': value_counting})

values
drop_nom_data = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

drop_ord_data = ['ord_5']

#train['ord_4'].value_counts()
## train set 정리

dr_train = train.drop(drop_nom_data, axis=1)

new_train = dr_train.drop(drop_ord_data, axis=1)

## test set 정리

dr_test = test.drop(drop_nom_data, axis=1)

new_test = dr_test.drop(drop_ord_data, axis=1)



new_train
## train, test set categorical column 추출

new_cat_col = find_category_col(new_train)

new_cat_col
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor



label_train = new_train.copy()

label_test = new_test.copy()



encoder = LabelEncoder()



for cols in new_cat_col:

    label_train[cols] = encoder.fit_transform(new_train[cols])

    label_test[cols] = encoder.transform(new_test[cols])

label_test['ord_4']
x = label_train.drop('target', axis=1)

y = label_train['target']



x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=7)
random_regressor = RandomForestRegressor(n_estimators=100, max_leaf_nodes=8)

random_regressor.fit(x_train, y_train)



y_check = random_regressor.predict(x_val)



#tree_regressor = DecisionTreeRegressor(n_estimators=100, max_leaf_nodes=16)
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=4)

xgb_model.fit(x_train, y_train, 

             early_stopping_rounds=10, 

             eval_set=[(x_val, y_val)], 

             verbose=False)

xgb_pred = xgb_model.predict(x_val)

print(roc_auc_score(y_val, xgb_pred))

mean_absolute_error(y_val, xgb_pred)
xgb_test_pred = xgb_model.predict(label_test)
from sklearn.datasets import load_iris

from sklearn.linear_model import Perceptron



iris = load_iris()



per_clf = Perceptron()

per_clf.fit(x_train, y_train)



pcp_pred = per_clf.predict(x_val)

roc_auc_score(y_val, pcp_pred)
pcp_test_pred = per_clf.predict(label_test)
id = label_test['id']
submission = pd.DataFrame({'id': id, 'target': xgb_test_pred})

#submission['target'].value_counts()
submission
submission.to_csv('submission3.csv', index=False)
train
## bin_0 : 거의 비슷하다

train['bin_0'].value_counts()

plt.figure(figsize=(10,10))

sns.barplot(x=train.bin_0, y=train.target)
## bin_1

train['bin_1'].value_counts() # '0' >> '1'

#plt.figure(figsize=(20,10))

sns.barplot(x=train.bin_1, y=train.target) # '0'값을 가진 항목이 '1'의 비율을 더 많이 가짐
##bin_2

train['bin_2'].value_counts() #'0':185000, '1':115000

sns.barplot(x=train['bin_2'], y=train['target']) #'0'과 '1'이 target에 미치는 영향은 거의 비슷(1이 더 우세하다)
##bin_3

train['bin_3'].value_counts() # 'T':153535  'F':146465

sns.catplot(data=train, x='bin_3', y='target', hue='bin_0', kind='bar', palette='dark', alpha=.5, height=7)

## bin_3('T'): bin_0('0') >> bin_0('1')  

## bin_3('F'): bin_0('0') >> bin_0('1')

##  ===> bin_3이 T일 때는 bin_0가 0,  bin_3이 F일 때는 bin_0가 1일 때 target이 1이 될 확률 up

sns.catplot(data=train, x='bin_3', y='target', hue='bin_1', kind='bar', palette='dark', alpha=.7, height=7)

## bin_3('T'): bin_1('0') >> bin_1('1')  

## bin_3('F'): bin_1('0') >> bin_1('1')

##  ===> bin_3이 T, F일 때는 bin_1이 0이면 target이 1이 될 확률 up

sns.catplot(data=train, x='bin_3', y='target', hue='bin_2', kind='bar', ci='sd', palette='dark', alpha=.7, height=7)

## bin_3('T'): bin_2('1') >> bin_2('0')  

## bin_3('F'): bin_2('1') >> bin_2('0')

##  ===> bin_3이 T, F일 때는 bin_2이 1이면 target이 1이 될 확률 up

plt.show()
##bin_4

train['bin_4'].value_counts() #'Y':191633, 'N':108367

sns.barplot(x=train.bin_4, y=train.target) #N이 Y보다 target에서 1의 비율이 조금 더 많음

sns.catplot(data=train, x='bin_4', y='target', hue='bin_3', kind='bar')

## bin_4('Y'): bin_3('F') >> bin_3('T')  

## bin_4('N'): bin_3('F') >> bin_3('T')

##  ===> bin_4이 Y, N일 때는 bin_3이 F이면 target이 1이 될 확률 up
values
values.iloc[11,1]

values.loc[11,'values count']
#train.nom_9.value_counts() <==버려
train.target.value_counts()