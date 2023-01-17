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

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
dataset=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
dataset.head()
dataset.shape
dataset.isnull().sum()
sns.heatmap(dataset.isnull(),cmap='viridis')
numerical_features=[feature for feature in dataset.columns if dataset[feature].dtypes!='O']
numerical_features
dataset.age_approx.isnull().sum()
dataset['age_approx'].describe()
max=60
dataset['age_approx']=dataset['age_approx'].fillna(max)
dataset['age_approx'].isnull().sum()
categorical_features=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']
categorical_features
dataset[categorical_features].head()
dataset[categorical_features].isnull().sum()
dataset[categorical_features].sex.describe()
dataset['sex']=dataset['sex'].fillna('male')
dataset[categorical_features].isnull().sum()
dataset['anatom_site_general_challenge'].describe()
dataset['anatom_site_general_challenge']=dataset['anatom_site_general_challenge'].fillna('torso')
dataset[categorical_features].columns
for feature in dataset[categorical_features]:
    print()
    print(dataset[feature].unique())
    print()
    print(len(dataset[feature].unique()))
dataset.drop(['diagnosis','benign_malignant'],axis=1,inplace=True)
dataset.head()
len(dataset['patient_id'].unique())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for feature in ['image_name', 'patient_id', 'sex', 'anatom_site_general_challenge']:
    dataset[feature]=le.fit_transform(dataset[feature])
dataset.head()
test=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
test.head()
len(test['patient_id'].unique())
test.shape
test.isnull().sum()
test['anatom_site_general_challenge'].describe()
test['anatom_site_general_challenge']=test['anatom_site_general_challenge'].fillna('torso')
test.isnull().sum()
dataset.head()
X=dataset.drop(['target'],axis=1)
y=dataset['target']
y.head()
y=pd.DataFrame(data=y)
y.head()
test.head()
test.columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for feature in ['image_name', 'patient_id', 'sex','anatom_site_general_challenge']:
    test[feature]=le.fit_transform(test[feature])
test.head()
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X,y,random_state=0)
y_train
from sklearn.neighbors import KNeighborsClassifier
KNC=KNeighborsClassifier()
KNC.fit(X_train,y_train)
test_pred=KNC.predict(test)
sub=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
sub.loc[:,'target']=test_pred
sub.to_csv('submission.csv',index=False)
