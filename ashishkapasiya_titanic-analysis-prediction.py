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
import pandas as pd
import numpy as np
# Library for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#ignore warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Load in the train and test datasets
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
# Store our passenger ID for easy access
PassengerId = test['PassengerId']
train.head()
print(train.shape)
print(test.shape)
titanic = pd.concat([train,test],keys=('train','test'))
titanic
titanic.head(180)
titanic.info()
pd.set_option('display.float_format', '{:.2f}'.format) # to change the format from 0.0000 to 0.00
corrmat = titanic.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,8))
#plot heat map
g=sns.heatmap(titanic[top_corr_features].corr(),annot=True,cmap="RdYlGn")
titanic['Age'].hist(bins=60) ## normally disturbuted
titanic['Fare'].hist(bins=40) ## right skewed
sns.boxplot(titanic['Age'])
sns.boxplot(titanic['Fare'])
# Age
Upper_boundary = titanic.Age.mean() + 3* titanic.Age.std()
Lower_boundary = titanic.Age.mean() - 3* titanic.Age.std()
# Fare
IQR = titanic.Fare.quantile(0.75) - titanic.Fare.quantile(0.25)
Lower_fence = titanic.Fare.quantile(0.25) - (IQR * 3)
Upper_fence = titanic.Fare.quantile(0.75) + (IQR * 3)
print('value of age',Upper_boundary , Lower_boundary)
print('value of fare',Upper_fence , Lower_fence )
titanic.duplicated().sum()
titanic.isnull().sum()
titanic.groupby('SibSp').Age.mean()
titanic['Age']=np.where((titanic['SibSp']==0)& (titanic['Age'].isnull()),30.0,titanic['Age'])
titanic['Age']=np.where((titanic['SibSp']==1)& (titanic['Age'].isnull()),31.0,titanic['Age'])
titanic['Age']=np.where((titanic['SibSp']==2)& (titanic['Age'].isnull()),23.0,titanic['Age'])
titanic['Age']=np.where((titanic['SibSp']==3)& (titanic['Age'].isnull()),16.0,titanic['Age'])
titanic['Age']=np.where((titanic['SibSp']==4)& (titanic['Age'].isnull()),9.0,titanic['Age'])
titanic['Age']=np.where((titanic['SibSp']==5)& (titanic['Age'].isnull()),10.0,titanic['Age'])
titanic['Age']=np.where((titanic['SibSp']==8)& (titanic['Age'].isnull()),15.0,titanic['Age'])
titanic['Embarked']=titanic['Embarked'].fillna(titanic['Embarked'].mode()[0]) 
titanic['Cabin']= titanic.Cabin.str.split('',expand=True)[1]
titanic.Cabin # to etract the value which is related to data
titanic.Cabin.fillna(method='ffill',inplace=True)
titanic['Alone'] =  titanic.Parch + titanic.SibSp
titanic['Alone'].loc[titanic['Alone'] >0] = 'With Family'
titanic['Alone'].loc[titanic['Alone'] == 0] = 'Alone'
titanic[['First','Last']] =titanic.Name.str.split(',',expand=True) 
titanic['Title']=titanic.Last.str.split('.',expand=True)[0]
titanic.Title.value_counts()
titanic['Title'] = titanic['Title'].replace(to_replace=['Dr','Rev','Mile','Col','Major','Lady','Jonkheer','Ms','Mlle','the Countess','Don','Mme','Capt','Sir'],value='Other',regex=True)
titanic['Person']=pd.cut(titanic['Age'], bins=[0,9,18,30,50,99], labels=['Child','Student','Young adult','Adult','Old'])
titanic['Fare_new']=pd.cut(titanic['Fare'], bins=[-1,120,250,380,600], labels=['Low','Medium','Average','High'])
titanic[titanic.Fare_new.isnull()]
titanic.isnull().sum()
titanic.drop(['First','Last','Ticket','Name','PassengerId','Age','Fare'],1,inplace=True)
titanic
titanic.Cabin.fillna('C',inplace=True)
titanic.Fare_new.fillna('Low',inplace=True)
df_num = titanic[titanic.select_dtypes(include = np.number).columns]
df_cat = titanic[titanic.select_dtypes(include = 'object').columns]
df_dummy = pd.get_dummies(df_cat,drop_first=True)
df = pd.concat([df_num,df_dummy],1)
df
data = df.loc[pd.IndexSlice[['train'],::]]
x_train =data.drop('Survived',1)
y_train = data.Survived
data1 = df.loc[pd.IndexSlice[['test'],::]]
x_test = data1.drop('Survived',1)
model = LogisticRegression() 
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
Submission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': y_pred })
Submission


















