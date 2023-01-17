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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
# missing data depcition 



train.isnull()
#EDA 



# show missing data 

sns.heatmap(train.isnull(),yticklabels=False,cbar=False)

# by looking at map below , concluse thata  lot of age information and cabin information is  missing 
# Ratio of who surviced and with didnt 



sns.countplot(x='Survived',data=train)
# hue for male female from the survided and non survived 

sns.countplot(x='Survived',hue='Sex',data=train)



# notice males mostly didnt survice in survive 0 

# females mostly in survived 1 
# from those who surviced  , diff by class

sns.countplot(x='Survived',hue='Pclass',data=train)



# notice the non survivied rate is highed for class 3 probaly cheapest and most unlikely to be survived in case of disaster

# from those survived , class 1 performed best
# Age distribution of passengers , drop the na 



sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
train.info()
# distribution of siblings 

sns.countplot(x='SibSp',data=train)



# notice most people did not have siblings or spouse ie mostly single

# second highest had one relation sublings / spouse 
# distribution of fares distribution 



train['Fare'].hist(color='green',bins=40,figsize=(8,4))



# motsly sold were cheapest tickets 
import cufflinks as cf

cf.go_offline()

train['Fare'].iplot(kind='hist',bins=30,color='green')
# Imputation , filling in missing age 



# distrubution of age in class 1 , 2 and 3

plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')



# looks like avg age for class 1 is aroudn 37 ,

# avg age for class 2 is 29

# avg age for class 3 24



# all null ages in these classes can be sustituted 
# make function to substitue age 



def impute_age(cols):

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

    



train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
# notice that the age is now filed in heatap for is null



sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
# either impute data for cabin not m[possible too much missing information , hence drop data]

train.drop('Cabin',axis=1,inplace=True)

train.head()
#final heatmap 



sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
# Convert catagorical features into dummy variables / indicators 



# we can have either male female or we can succesfully trubcate this to be one column

pd.get_dummies(train['Sex'])
sex = pd.get_dummies(train['Sex'],drop_first=True)

sex
# Simillarly do for embard



embark = pd.get_dummies(train['Embarked'],drop_first=True)

embark
# drop old columns , name and ticket and not used in survival 

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)



# concat new ones to dataframe 

train = pd.concat([train,sex,embark],axis=1)



train.head()
train.tail()
# passengerId is also not a useful metric , so drop that too 

# train.drop(['PassengerId'],axis=1,inplace=True)

train.nunique()
# Building a Logistic Regression model



# Mark X and Y 



# X is everything ebsides survices 

X = train.drop("Survived",axis=1)



# Y is surviced flag 

Y = train["Survived"]



print(X)
# train test split 



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y , test_size=.469135802469, random_state=101)
from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
X_test
predictions = logmodel.predict(X_test)
# Evaluation 



from sklearn.metrics import classification_report



print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix



confusion_matrix(y_test,predictions)
# Increase accuracy tip



# grab title of name and use as fetaure

# grab ticket value 
train.PassengerId
len(predictions)
output = pd.DataFrame({'PassengerId': X_test.PassengerId , 'Survived': predictions})

output.to_csv('titanic_survived_submission_47.csv', index=False)

print("Your submission was successfully saved!")