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
gender_sub=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
# gender_sub.drop('PassengerId',axis=1,inplace=True)
gender_sub.head()
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.describe()
tr_data=pd.read_csv('/kaggle/input/titanic/train.csv')
tr_data.groupby(['Survived']).mean()
tr_data.head()
tr_data['Age'].fillna(tr_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
# tr_data['Age'].mean()
tr_data['Embarked']=tr_data['Embarked'].fillna('S')
tr_data.isnull().sum()
# tr_data.head()
# tr_data.columns.unique()
tr_data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
test_data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
test_data.head()
tr_data['Sex']=tr_data['Sex'].map({'male':0,'female':1})
test_data['Sex']=test_data['Sex'].map({'male':0,'female':1})
tr_data['Cabin_indexed']=np.where(tr_data['Cabin'].isnull(),0,1)
test_data['Cabin_indexed']=np.where(test_data['Cabin'].isnull(),0,1)
tr_data.drop(['Cabin'],axis=1,inplace=True)
test_data.drop(['Cabin'],axis=1,inplace=True)
tr_data.groupby('Survived').mean()
from sklearn.preprocessing import OneHotEncoder

# Get list of categorical variables
s = (tr_data.dtypes == 'object')
object_cols = list(s[s].index)
# object_cols
OH_Enc=OneHotEncoder(handle_unknown='ignore',sparse=False)

OH_enc_tr=pd.DataFrame(OH_Enc.fit_transform(tr_data[object_cols]))
OH_enc_test=pd.DataFrame(OH_Enc.fit_transform(test_data[object_cols]))

OH_enc_tr.index=tr_data.index
OH_enc_test.index=test_data.index

OH_num_tr=tr_data.drop(object_cols,axis=1)
OH_num_test=test_data.drop(object_cols,axis=1)

OH_X_tr=pd.concat([OH_num_tr,OH_enc_tr],axis=1)
OH_X_test=pd.concat([OH_num_test,OH_enc_test],axis=1)
# OH_X_tr
# OH_X_test

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# label_X_train = label_encoder.fit_transform(tr_data['Embarked'])
OH_X_test.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
for i, col in enumerate(['Pclass','Sex','SibSp','Parch','Embarked','Cabin_indexed']):
    plt.figure(i)
    sns.catplot(x=col,y='Survived',data=tr_data,kind='point',aspect=2)
tr_data['Family_mem']=tr_data['SibSp']+tr_data['Parch']
tr_data.drop(['SibSp','Parch'],axis=1,inplace=True)
test_data['Family_mem']=test_data['SibSp']+test_data['Parch']
test_data.drop(['SibSp','Parch'],axis=1,inplace=True)
test_data.head()
for i, col in enumerate(['Pclass','Sex','Embarked','Cabin_indexed','Family_mem']):
    plt.figure(i)
    sns.catplot(x=col,y='Survived',data=tr_data,kind='point',aspect=2)
tr_data.pivot_table('Survived',index='Sex',columns='Embarked',aggfunc='count')
OH_X_tr.to_csv('tr_X_data_cl.csv',index=False)
OH_X_test.to_csv('test_X_data_cl.csv',index=False)

tr_X_data_cl=pd.read_csv('tr_X_data_cl.csv')
test_X_data_cl=pd.read_csv('test_X_data_cl.csv')
tr_x_data=tr_X_data_cl.drop(['Survived'],axis=1)
tr_y_data=tr_X_data_cl['Survived']

test_x_data=test_X_data_cl
test_x_data['Fare']=test_x_data.fillna(method='bfill',axis=0)
test_x_data.isnull().sum()
# tr_y_data
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(tr_x_data,tr_y_data)
pred_y=rf.predict(test_x_data)
gender_sub.drop(['PassengerId'],axis=1,inplace=True)
gender_sub.head()
pred_y=pd.DataFrame(pred_y,columns=['Pred_Survived'])
pred_y.loc[pred_y['Pred_Survived'] > 0.5,'Pred_Survived'] = 1
pred_y.loc[pred_y['Pred_Survived'] < 0.5,'Pred_Survived'] = 0
# if pred_y['Pred_Survived'].any()<=0.5:
#     pred_y['Pred_Survived'].set_values==0
# elif pred_y['Pred_Survived'].values.any()>0.5:
#     pred_y['Pred_Survived'].values=1
pred_y.head()
from sklearn.metrics import mean_absolute_error
gender_sub=gender_sub
score=mean_absolute_error(pred_y,gender_sub)
score
pred_y['Gender_sub']=gender_sub
pred_y.sum()
