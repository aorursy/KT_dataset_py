# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
tdf = pd.read_csv("../input/train.csv")
tdf.head()
tdf.info()
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.countplot('Sex', data =tdf,hue='Survived' )
# Show relationshio between class and Survival - shows that passengers in class-3 lowest class had major casualities
sns.countplot('Pclass', data = tdf, hue = 'Survived')
#get average age class wise to fill in missing age value
sns.boxplot(x='Pclass', y='Age', data = tdf)
sns.heatmap(tdf.isnull(), yticklabels=False, cbar = False)
# heatmap and tdf.info() shows that almost 20% values are missing for Age, almost 80% values are missing for Cabin and very few for Embarked
tdf['Age'].mean()
tdf[tdf['Pclass'] == 1]['Age'].dropna().mean()
tdf[tdf['Pclass'] == 2]['Age'].dropna().mean()
tdf[tdf['Pclass'] == 3]['Age'].dropna().mean()
def setAge(cols):
    age = cols[0]
    pclass = cols[1]
        
    if pd.isnull(age):
        if (pclass == 1):
            return 38.23
        elif (pclass == 2):
            return 29.87
        elif (pclass ==3):
            return 25.14
    else:
        return age
                
tdf['Age'] = tdf[['Age','Pclass']].apply(setAge, axis = 1)
sns.heatmap(tdf.isnull(), yticklabels=False,cbar=False)
# check if and Age values are still null
# As we are missing 80% of Cabin values, we can drop Cabin column
tdf.drop('Cabin', axis=1, inplace=True)
tdf.head()
tdf.info()
#still 2 values (891 - 889) are null for column - Embarked. as the number is very small as compared to total dataset, we will drop those values
tdf.dropna( inplace=True)
tdf.info()
# the 2 null values are dropped and all columns have same number of elements
# convert text categorical columns to integer columns

gender = pd.get_dummies(tdf['Sex'], drop_first=True)
gender.head()
embarked = pd.get_dummies(tdf['Embarked'], drop_first=True)
embarked.head()
pclass = pd.get_dummies(tdf['Pclass'], drop_first=True)
pclass.head()
titanicdf = pd.concat([tdf,gender, pclass, embarked] , axis = 1)
titanicdf.head()
#drop non numeric columns and PassengerId column as it will not be useful for prediction

titanicdf.drop(['Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace=True)
titanicdf.info()
from sklearn.model_selection import train_test_split
y = titanicdf['Survived']
X = titanicdf

X.drop(['Survived'], axis = 1, inplace= True)

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300, max_leaf_nodes= 50)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
from sklearn import metrics
print(metrics.confusion_matrix(y_test, rfc_pred))
print(metrics.classification_report(y_test, rfc_pred))
from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(y_test, rfc.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
print('ROC AUC {}'.format(roc_auc))
