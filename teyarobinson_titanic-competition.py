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
#gender csv into dataframe

gender_df = pd.read_csv('../input/gender_submission.csv')

#gender_df
#train data into dataframe

train_data_df = pd.read_csv('../input/train.csv')

train_data_df
#How many males vs. females?

train_data_df['Sex'].value_counts()

#How many 1st, vs. 2nd, vs. 3rd class?

train_data_df['Pclass'].value_counts()
#age info 

#train_data_df['Age'].value_counts().sort_index()

train_data_df.fillna(0)
#train_data_df['Age'].value_counts().sort_index()
#How many survived? 

train_data_df['Survived'].value_counts()
train_data_df.dtypes
train_data_df.Sex.replace(to_replace=dict(female=1, male=0), inplace=True)
#test data into dataframe

test_data_df = pd.read_csv('../input/test.csv')

#test_data_df
#Logistics Model below
# Assign X (data) and y (target)

X = train_data_df[['PassengerId','Pclass', 'Sex']]

y = train_data_df["Survived"]

print(X.shape, y.shape)
#from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
#create a logistics regression model 

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier
classifier.fit(X_train, y_train)
print(f"Training Data Score: {classifier.score(X_train, y_train)}")

print(f"Testing Data Score: {classifier.score(X_test, y_test)}")
predictions = classifier.predict(X_test)

print(f"First 10 Predictions:   {predictions[:10]}")

print(f"First 10 Actual labels: {y_test[:10].tolist()}")
pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)