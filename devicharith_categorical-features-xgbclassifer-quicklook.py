import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sn



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv',index_col = 'id')

test_data = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col = 'id')

print(train_data.shape,test_data.shape)
study_train_data = pd.DataFrame({'columns':train_data.columns})

study_train_data['datatypes'] = train_data.dtypes.values

study_train_data['missing'] = train_data.isnull().sum().values

study_train_data['unique'] = train_data.nunique().values



print(study_train_data)



                                
number = test_data.isnull().sum().values



missing_values_df = pd.DataFrame({'coloumns': test_data.columns,

                                   'missing_value': number })



fig, ax = plt.subplots(figsize=(17, 6))

g = sn.barplot(x='coloumns',y = 'missing_value' ,data = missing_values_df,palette ='spring')



#for p in g.patches:

    #g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.show()
info_test = pd.DataFrame({'columns': test_data.columns})

info_test['datatype'] = test_data.dtypes.values

info_test['missing_values'] = test_data.isnull().sum().values

info_test['unique_values'] = test_data.nunique().values



print(info_test)
fig ,ax = plt.subplots(figsize=(15,7))

sn.barplot(x = 'columns',y = 'missing_values',ax=ax,data =info_test,palette = 'spring')

plt.show()
train_cols_missing_values = [col for col in train_data.columns if train_data[col].isnull().any()]

test_cols_missing_values = [cols for cols in test_data.columns if test_data[cols].isnull().any()]



from sklearn.impute import SimpleImputer

impu = SimpleImputer(strategy='most_frequent')

train_data[train_cols_missing_values] = impu.fit_transform(train_data[train_cols_missing_values])

test_data[test_cols_missing_values] = impu.fit_transform(test_data[test_cols_missing_values])
fig, ax = plt.subplots(figsize = (5,7))

g = sn.countplot(x='target',ax = ax,data = train_data,palette ='spring')

for p in g.patches:

    g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



plt.show()
x_train = train_data.iloc[:, :-1]

y_train = train_data.iloc[:, -1]

x_test = test_data.iloc[:, ]



from sklearn.preprocessing import LabelEncoder

imput = LabelEncoder()

categorical_columns = [col for col in x_train.columns if x_train[col].dtype == 'object']

cate_col = [cols for cols in x_test.columns if x_test[cols].dtype == 'object']

for i in categorical_columns:

    x_train[i] = imput.fit_transform(x_train[i])

for j in cate_col:

    x_test[j] = imput.fit_transform(x_test[j])

print(x_train)

print(x_test)
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

x_train = scale.fit_transform(x_train)

x_test = scale.fit_transform(x_test)
from xgboost import XGBClassifier

classifer = XGBClassifier(booster = 'gbtree',eta = 0.1,max_depth = 10,objective = 'binary:logistic',eval_metric='auc',random_state = 2020)

classifer.fit(x_train,y_train)
y_pred = classifer.predict_proba(x_test)[:, 1]

print(y_pred)
submission_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')

sub_id = submission_df['id']

submission = pd.DataFrame({'id':sub_id})

submission['target'] = y_pred

submission.to_csv("submission5.csv",index = False)