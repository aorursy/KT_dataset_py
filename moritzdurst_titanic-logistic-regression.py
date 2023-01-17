# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plots

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





    # Any results you write to the current directory are saved as output.
# load data

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test  = pd.read_csv("/kaggle/input/titanic/test.csv")
# overview of the dataframe

train.head()
train.info()
# look for any missing data

# as the heatmap shows, there are empty cells (yellow) in the 'Age' and 'Cabin' columns

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Countplot for the outcome variable 'Survived'

# number of observations for the outcome variable 'Survived' as a function of 'Sex'

sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='rainbow')
# Countplot for the outcome variable 'Survived'

# number of observations for the outcome variable 'Survived' as a function of 'Pclass'

sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
# Distribution of 'Age'

sns.distplot(train['Age'].dropna(),kde=False,color='blue',bins=30)
# Empty cells in 'Age' can be filled so they respective row does not have to be deleted for the model fit

# One way to fill these empty cells is by imputing the median 'Age' of all passengers of the respective 'Pclass'



# How is 'Age' distributed for each 'Pclass'? --> Boxplot

# Passengers in the more expensive classes tend to be older than in the lower classes

plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='rainbow')

# This function will fill the empty cells in our dataframe:

def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age): # if the respective cell is empty, impute it with the...



        if Pclass == 1:

            return 37 # ... median 'Age' for 'Pclass' == 1



        elif Pclass == 2:

            return 29 # ... median 'Age' for 'Pclass' == 2



        else:

            return 24 # ... median 'Age' for 'Pclass' == 3



    else:

        return Age
# Empty cells in column 'Age' are being imputed, using the new function

# Function accesses the columns 'Age' and 'Pclass'

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
# let's see if there are still empty cells in our dataframe:

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# column 'cabin' contains many empty cells --> this column is now be deleted

train.drop('Cabin',axis=1,inplace=True)
# categorical variables are transformed into dummy variables (0s and 1s) 

sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)



# 'Name' and Ticket' cannot be transformed and are thus dropped in the next step
# all categorical variables are deleted from the dataframe

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
# transformed (dummy-coded) columns of previously categorical variables are added to the train dataframe

train = pd.concat([train,sex,embark],axis=1)
train.head()
# Dataframe needs to be split into...

# ... training data --> to fit the model

# ... test data --> to test the model

from sklearn.model_selection import train_test_split
# Dataframes for input variables (X) and output variable (y) are created

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=99)
# logistic regression model is imported

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

# Logistic Regression model is trained

logmodel.fit(X_train,y_train)
# Logistic Regression Model is used to predict the outcome variable (y) in the test dataframe...

# ... based on the input variables in the test dataframe

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
# Model Fit Evaluation

print(classification_report(y_test,predictions))