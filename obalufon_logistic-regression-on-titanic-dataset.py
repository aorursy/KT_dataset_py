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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()
#Summary statistics

train_df.describe()
train_df.info()
#checking for missing data 

sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='plasma')
train_df['Age'].hist(bins=30,color='blue',alpha=0.7)
sns.countplot(x='Survived',data=train_df,palette='Greens')
#Average age by passenger using boxplot

plt.figure(figsize=(10, 7))

sns.boxplot(x='Pclass',y='Age',data=train_df,palette='magma')
#function to fill average age for each passenger class

def ageFiller(cols):

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
train_df['Age'] = train_df[['Age','Pclass']].apply(ageFiller,axis=1)

sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='plasma')
train_df.drop('Cabin',axis=1,inplace=True)
train_df.drop('Name',axis=1,inplace=True)
train_df.drop('Ticket',axis=1,inplace=True)
train_df.head()
#Creating dummy variables

#Make sure to avoid dummy variable trap

sex = pd.get_dummies(train_df['Sex'],drop_first=True)

embark = pd.get_dummies(train_df['Embarked'],drop_first=True)

train_df.drop(['Sex','Embarked'],axis=1,inplace=True)

df = pd.concat([train_df,sex,embark],axis=1)
df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), 

                                                    df['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))