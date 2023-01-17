# basic ds imports

import numpy as np 

import pandas as pd



# visualization

import seaborn as sns

import plotly

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
# import data

train_df = pd.read_csv('../input/train.csv')  # training dataframe

test_df  = pd.read_csv('../input/test.csv')   # test dataframe
train_df.head()
# Let's look part of survived people and overall amount of male and female passengers

figure = plt.figure(figsize=(18,10))

figure.add_subplot(1,2,1)

train_df['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%')

figure.add_subplot(1,2,2)

train_df['Sex'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%')

plt.show()
from sklearn import preprocessing



# predict Embarked null values are 'S'

train_df['Embarked'].fillna('S',inplace=True)

# one-hot-encoding for sex and embarked (in other words text->numbers)

label_encoder = preprocessing.LabelEncoder()

train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])

train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])
train_df.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics #accuracy measure





# Lets use Class, Sex and Embarked to predict weather person is Survived or not

train, test = train_test_split(train_df, test_size=0.3,random_state=0)

target_col = ['Pclass', 'Sex', 'Embarked']





train_X=train[target_col]

train_Y=train['Survived']

test_X=test[target_col]

test_Y=test['Survived']



features_one = train_X.values

target = train_Y.values



# Use Logistic regression model to learn data

model = LogisticRegression()

model.fit(features_one, target)

dt_prediction = model.predict(test_X)



print('The accuracy of the Logistic Regression is',metrics.accuracy_score(dt_prediction, test_Y))
# Now test Sex, Embarked from text -> numbers

label_encoder = preprocessing.LabelEncoder()

test_df['Sex'] = label_encoder.fit_transform(test_df['Sex'])

test_df['Embarked'] = label_encoder.fit_transform(test_df['Embarked'])



# predict test data with pre-trained logistic regressin model

test_features = test_df[target_col].values

dt_prediction_result = model.predict(test_features)



# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId = np.array(test_df["PassengerId"]).astype(int)

dt_solution = pd.DataFrame(dt_prediction_result, PassengerId, columns = ["Survived"])



# Write your solution to a csv file with the name baseline_solution.csv

dt_solution.to_csv("baseline_solution.csv", index_label = ["PassengerId"]) 