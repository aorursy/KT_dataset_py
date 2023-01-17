# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
my_data = pd.read_csv('../input/train.csv')

# Any results you write to the current directory are saved as output.
my_data.describe()
my_gender = pd.read_csv('../input/train.csv')

my_data.info()
my_gender.info()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
my_data.head()
sns.heatmap(my_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=my_data,palette='rainbow')
sns.distplot(my_data['Age'].dropna(),kde=False,color='darkred',bins=30)
my_data['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=my_data,palette='winter')
def age_replace(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
my_data['Age']=my_data[['Age','Pclass']].apply(age_replace,axis=1)
my_data.drop('Cabin',axis=1,inplace=True)
sex = pd.get_dummies(my_data['Sex'],drop_first=True)
embark = pd.get_dummies(my_data['Embarked'],drop_first=True)
my_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
my_data = pd.concat([my_data,sex,embark],axis=1)
my_data.head()
class_p = pd.get_dummies(my_data['Pclass'],drop_first=True)
my_data = pd.concat([my_data,class_p],axis=1)
my_data.head()
my_data.drop(['Pclass'],inplace=True,axis=1)
my_data.head()
x= my_data.drop('Survived',axis=1)
y=my_data['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, 
                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
