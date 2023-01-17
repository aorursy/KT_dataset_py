# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import shap

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/titanic/train.csv')
# Creating initials for imputing  Null Values of Age

data['Initial']=0

for i in data:

    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')
# Replacing missspelled Initials

data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
data.groupby('Initial')['Age'].mean()
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33

data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36

data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5

data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22

data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46
data.Age.isnull().any()
data['Embarked'].fillna('S',inplace=True)
data.Embarked.isnull().any()
data['Age_band']=0

data.loc[data['Age']<=16,'Age_band']=0

data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1

data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2

data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3

data.loc[data['Age']>64,'Age_band']=4
data['Family_Size']=0

data['Family_Size']=data['Parch']+data['SibSp']#family size

data['Alone']=0

data.loc[data.Family_Size==0,'Alone']=1
data['Fare_cat']=0

data.loc[data['Fare']<=7.91,'Fare_cat']=0

data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1

data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2

data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3
data['Sex'].replace(['male','female'],[0,1],inplace=True)

data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
ignore_cols= ['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId','Survived']
sel_cols = [col for col in data.columns if col not in ignore_cols]

sel_cols
target = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.33, random_state=42,stratify=target)
model = RandomForestClassifier(n_estimators=200,random_state=0)

model.fit(X_train[sel_cols],y_train)

prediction=model.predict_proba(X_test[sel_cols])

print('Roc-auc score is',metrics.roc_auc_score(y_test,prediction[:,1]))
X_test.iloc[2,:]
explainer = shap.TreeExplainer(model)

row_to_show = 2

data_for_prediction = X_test[sel_cols].iloc[row_to_show]

shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
X_test.iloc[0,:]
row_to_show = 0

data_for_prediction = X_test[sel_cols].iloc[row_to_show]

shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
!pip install shapwaterfall
from shapwaterfall import shapwaterfall
X_test.rename(columns={'Name' : 'Reference'},inplace=True)
X_test.head()
pred_df = pd.DataFrame({'Reference': X_test.Reference,'prediction': prediction[:,1]})
pred_df.head()
shapwaterfall(model,X_train[sel_cols],X_test[sel_cols+['Reference']],ref1 = 'Wiklund, Mr. Jakob Alfred',ref2 = 'West, Mrs. Edwy Arthur (Ada Mary Worth)', num_feature=5)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set(rc={'figure.figsize':(15,12)})

plt.subplot(2, 3, 1)

sns.countplot(data= data, x='Sex',hue='Survived')

plt.subplot(2, 3, 2)

sns.countplot(data= data, x='Initial',hue='Survived')

plt.subplot(2, 3, 3)

sns.countplot(data= data, x='Pclass',hue='Survived')

plt.subplot(2, 3, 4)

sns.countplot(data= data, x='Parch',hue='Survived')

plt.subplot(2, 3, 5)

sns.countplot(data= data, x='Family_Size',hue='Survived')


