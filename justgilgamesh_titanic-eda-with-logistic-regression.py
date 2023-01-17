# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
%matplotlib inline
# read csv file
titanicdf = pd.read_csv('/kaggle/input/titanic/train.csv')
# check data using .head()
titanicdf.head()
# check information based on basic information
titanicdf.info()
# find the summary of the data
titanicdf.describe()
# check for missing data within the dataset
sns.heatmap(titanicdf.isnull(),yticklabels=False,cbar=False)
# drops the Cabin column
titanicdf = titanicdf.drop(axis=1, columns=['Cabin'])
sns.countplot(x='Survived',data=titanicdf,palette='Paired')
titanicdf[['Pclass', 'Survived']].groupby('Pclass').mean().sort_values(by='Survived', ascending=False)
sns.countplot(x='Survived',hue='Pclass',data=titanicdf,palette='Paired')
titanicdf[['Sex', 'Survived']].groupby('Sex').mean()
sns.countplot(x='Survived',hue='Sex',data=titanicdf,palette='Paired')
titanicdf[['Survived', 'Embarked']].groupby('Embarked').mean()
sns.countplot(x='Survived',hue='Embarked',data=titanicdf,palette='Paired')
# for the missing age values, we take the mean of each Pclass age values
titanicdf[['Pclass','Age']].groupby('Pclass').mean()
plt.figure(figsize=(12,6))
sns.boxplot(x='Pclass',y='Age',data=titanicdf, palette='Paired')
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 29

        else:
            return 25

    else:
        return Age
titanicdf['Age'] = titanicdf[['Age','Pclass']].apply(impute_age,axis=1)
# check on missing data
sns.heatmap(titanicdf.isnull(),yticklabels=False,cbar=False)
SAge = sns.FacetGrid(titanicdf, col='Survived', height=3, aspect=1.6)
SAge.map(plt.hist, 'Age', alpha=.7, bins=20)
SAPclass = sns.FacetGrid(titanicdf, col='Survived', row='Pclass', height=3, aspect=1.6)
SAPclass.map(plt.hist, 'Age', alpha=.7, bins=20)
SAPclass.add_legend();
sex = pd.get_dummies(titanicdf['Sex'],drop_first=True)
embark = pd.get_dummies(titanicdf['Embarked'],drop_first=True)
titanicdf.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
titanicdf = pd.concat([titanicdf, sex, embark], axis =1)
titanicdf.head()
X_train, X_test, y_train, y_test = train_test_split(titanicdf.drop('Survived',axis=1), 
                                                    titanicdf['Survived'], test_size=0.30, 
                                                    random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
