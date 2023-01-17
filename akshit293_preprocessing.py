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
data=pd.read_csv('../input/titanic/train.csv')
data.head()
data.skew()
data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
data['Sex']=data['Sex'].replace(['male','female'],[1,0])
print(data['Sex'].value_counts())
#creating another feature
#treating sex feature as male and creating another feature female
data['Female']=data['Sex']^1
print(data['Female'].value_counts())
data['Embarked'].value_counts()
#removing nan values from Age and Embarked columns
data['Age']=data['Age'].replace(np.nan,data['Age'].median())
data['Embarked']=data['Embarked'].replace(np.nan,'S')
data.info()
data['Embarked']=data['Embarked'].replace(['S','C','Q'],[1,2,3])
print(data['Embarked'].value_counts())
#creating different columns for each port of embarkation
data['C']=data['Embarked']^1
data['C']=data['C'].replace([2,3],[0,1])
print(data['C'].value_counts())
data['Q']=data['Embarked']^1
data['Q']=data['Q'].replace([3,2],[0,1])
print(data['Q'].value_counts())
#treating embarked column as port of embarkation=S
data['Embarked']=data['Embarked'].replace([2,3],[0,0])
print(data['Embarked'].value_counts())
data['Pclass'].value_counts()
#treating pclass as pclass1
#creating different feature for each pclass
data['pclass2']=data['Pclass'].replace([3,1,2],[0,0,1])
data['pclass2'].value_counts()

data['pclass3']=data['Pclass'].replace([1,3,2],[0,1,0])
data['pclass3'].value_counts()
#removing value of all different pclass from pclass column as it now contains only pclass1
data['Pclass']=data['Pclass'].replace([2,3],0)
data['Pclass'].value_counts()
data.head()
y=pd.DataFrame(data['Survived'])
data.drop(['Survived'],axis=1,inplace=True)
data.info()