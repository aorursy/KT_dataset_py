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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Reading the data into a pandas dataframe. 
train = pd.read_csv('../input/train.csv')
# Exploratory data analysis using .info()
train.info()
# Missing data - from the above info about the dataframe it can be seen that there should be 891 values for each column. However, some of these such as 'age', 'cabin' and 'embarked' are not complete
# We can draw a plot using seaborn to visualise this.
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='cool')
# From this heatmap we can see 'cabin' has lots of data missing and 'age' has some data missing.
# Continuing exploratory data analysis 
sns.set_style('darkgrid')

sns.countplot(x='Survived', data=train)
# From this it can be seen that aprox 550 people did not survive, vs 330 aprox that survived. 
# The data can be subplotted taking into account gender. 
sns.countplot(x= 'Survived', hue='Sex', data=train, palette = 'muted')
# Showing the data like this shows the trend, people who didn't survive were much more likely to be male, and people who survived were much more likely to be female. 

sns.countplot(x= 'Survived', hue='Pclass', data=train, palette = 'muted')
# This graph shows us that the people who did not survive were mostly 3rd class. 
# Looking at the distribution of age throughout the data. 
sns.distplot(train['Age'].dropna(), kde=False, bins=30)
# Ages look to be around 20-30. 
# How much did people pay to be on board? Bearing in mind most people were in 3rd class. 
train['Fare'].hist(bins=50, figsize=(10,5))
# Cleaning the data ready for the machine learning algorithms 
# Using imputation to fill in the missing ages, using average age by passenger class
plt.figure(figsize=(11,8))
sns.boxplot(x='Pclass', y='Age', data = train)
# From this box plot we can deduce that passengers in 1st or 2nd class tend to be older than 3rd class.
# We can use these average age values to impute age. 
# We can create a function to achieve this.

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

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap= 'cool')
# This heatmap shows that the Age column no longer has any missing values. 
# The cabin column is still missing lots of information. 
train.drop('Cabin', axis=1, inplace=True)
# This cabin column has been dropped from the data. 
train.head()
train.info()
# We still have missing values for embarked, we can drop any other missing values. 
train.dropna(inplace=True)

train.info()
# All columns now have 889 rows of filled in data. 
# ML algorithm cannot take in catergorical variable, e.g. male / female. It needs a dummy variable format, 0 or 1 value. 
pd.get_dummies(train['Sex'])
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train,sex,embark], axis=1)
train.head(2)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.head()
train.drop('PassengerId', axis=1, inplace=True)
train.head()
# Building a logistic regression model 
# treating the train dataframe as if it was all of the data
# lose the survive column as that's what I am trying to predict
X = train.drop('Survived', axis=1)
y = train['Survived']
# From scikit-learn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# Logistic regression model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
#Evaluate the model 
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
# What does this mean? - from scikit learn website 
#The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

#The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

#The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.

#The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.

#The support is the number of occurrences of each class in y_true.

