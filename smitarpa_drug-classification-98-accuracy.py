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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

from scipy.stats import chi2

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Read given data

data = pd.read_csv("/kaggle/input/drug-classification/drug200.csv")
#Check if any missing values

print(data.isnull().sum())
## Analyze relationships to find importance of independent features

plt.figure()

sns.countplot(x='Drug', hue='Sex', data=data)

plt.show()

# Shows target is not imbalanced. Also not much differnce in drugs applicable for M/F



plt.figure()

sns.countplot(x='Drug', hue='BP', data=data)

plt.show()

#DrugC is given only for Low BP, DrugB is given only for High BP.



plt.figure()

sns.countplot(x='Drug', hue='Cholesterol', data=data)

plt.show()



# Check relationship of numerical features with Target variable

plt.figure()

sns.catplot(x='Drug', y='Na_to_K', data=data)

plt.show()



plt.figure()

sns.catplot(x='Drug', y='Age', data=data)

plt.show()
# Visualize relationship between two categorical vars

cross_table = pd.crosstab(index=data['Sex'], columns = data['BP'])

cross_table.plot(kind='bar', figsize=(8,8), stacked=True)



plt.figure()

sns.boxplot(x='Age', y='BP', data=data)

plt.show()
#Separate last 15 records to make predictions on unseen data, rest taken for training the model

tr_df = data.iloc[:185,:]

test_df = data.iloc[185:, :]

# Convert categorical features to numeric before modelling

tr_df['Sex'] = tr_df['Sex'].map({'M':2, 'F':1})

tr_df['Cholesterol'] = tr_df['Cholesterol'].map({'HIGH':2, 'NORMAL':1})

tr_df['BP'] = tr_df['BP'].map({'HIGH':3, 'LOW':1, 'NORMAL':2})

tr_df['Drug'] = tr_df['Drug'].map({'DrugY':5, 'drugC':3, 'drugX':4, 'drugA':1, 'drugB':2})



#Identify X & y

X = tr_df.drop(columns='Drug', axis=1)

y = tr_df['Drug']



# Split remaining training data data into train and test sets

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Verify all columns are numeric

x_train.info()
# Model Decision Tree Classifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

y_tr_pred = dt.predict(x_train)

y_pred = dt.predict(x_test)

#Validate model

score_tr = accuracy_score(y_train, y_tr_pred) 

score_tst = accuracy_score(y_test, y_pred)

print("training, test score : ", score_tr, score_tst)



# validate score through cross validation.

cvs = cross_val_score(DecisionTreeClassifier(random_state=1), X, y, cv=5)

print("Cross validation score : ", np.mean(cvs))

# model gives accuracy of 98%
# Make predictions for unseen test data 



#Convert cat to num

test_df['Sex'] = test_df['Sex'].map({'M':2, 'F':1})

test_df['Cholesterol'] = test_df['Cholesterol'].map({'HIGH':2, 'NORMAL':1})

test_df['BP'] = test_df['BP'].map({'HIGH':3, 'LOW':1, 'NORMAL':2})

#test_df['Drug'] = test_df['Drug'].map({'DrugY':5, 'drugC':3, 'drugX':4, 'drugA':1, 'drugB':2})



y_pred_test = dt.predict(test_df.drop(columns='Drug', axis=1))

test_df['Predicted Drug'] = y_pred_test

test_df['Predicted Drug'] = test_df['Predicted Drug'].map({5:'DrugY', 3:'drugC', 4:'drugX', 1:'drugA', 2:'drugB'})



test_df.to_csv("Drug200Classification-predicted.csv")

#Open this file and compare actual vs predicted results