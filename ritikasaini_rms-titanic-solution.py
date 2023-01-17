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

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df=pd.read_csv("/kaggle/input/TheUnsinkable.csv")
df.info()
df.describe()
df.head(5) #sibsp-->sibling spouse, parch---->no of parents,children, embarked--->port of embarkment
df.isnull() #since we can't view whole data so we use visualisation using seaborn
df.corr('pearson')

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#map
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#sns.set_style('whitegrid')
sns.countplot(x='survived',data=df)
sns.countplot(x='survived',hue='sex',data=df) #In seaborn, the hue parameter determines which column in the data frame should be used for colour encoding. 
sns.countplot(x='survived',hue='pclass',data=df)
sns.distplot(df['age'].dropna(),kde=False,color='darkred',bins=40) #kde=Kernel Density Estimates, bins are class intervals the total range of dataset is divides into
sns.countplot(x='sibsp',data=df)
sns.distplot(df['fare'])
sns.boxplot(x='pclass',y='age', data=df) #we find mean age based on different classes, for class1- 37, for class2- 30, for class3-25
def impute_age(cols):
    age=cols[0]           #1st column is age
    pclass=cols[1]        #2nd column is pclass
    
    if pd.isnull(age):
        if pclass==1:
            return 37
        if pclass==2:
            return 29
        if pclass==3:
            return 24
    else:                  #when age is not null
        return age
        
df['age']=df[['age','pclass']].apply(impute_age, axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.drop('deck',axis=1, inplace=True)
df.head(5)
embark=pd.get_dummies(df['embarked'],drop_first=True)
sex=pd.get_dummies(df['sex'],drop_first=True)
ad_male=pd.get_dummies(df['adult_male'],drop_first=True)
df.drop(['sex','embarked','who','embark_town','alive','alone','adult_male','class'],axis=1,inplace=True)
df.head()
df=pd.concat([df,sex,embark,ad_male],axis=1)
df.head()
df.drop(['survived'],axis=1).head()
df['survived'].head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test=train_test_split(df.drop('survived',axis=1),df['survived'],test_size=0.3,random_state=101)
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
pred=logmodel.predict(X_test)
accuracy=confusion_matrix(y_test,pred)
accuracy
accuracy=accuracy_score(y_test,pred)
accuracy
pred