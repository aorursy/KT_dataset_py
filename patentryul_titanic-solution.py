# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# data analysis

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# hide useless warnings...

import warnings

warnings.filterwarnings('ignore')



# import data to pandas instance from csv file

train_df = pd.read_csv('../input/train.csv')  # training dataframe

test_df  = pd.read_csv('../input/test.csv')   # test dataframe

train_df.head()
# show chart by seaborn // 1: Survived, 2: Died

f, ax=plt.subplots(1, 2, figsize=(18,8))

train_df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=train_df,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
# show chart by gender

f,ax=plt.subplots(1,2,figsize=(18,8))

train_df['Survived'][train_df['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

train_df['Survived'][train_df['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[0].set_title('Survived (male)')

ax[1].set_title('Survived (female)')

plt.show()
# show table by cabin class(use pandas function)

pd.crosstab([train_df['Sex'],train_df['Survived']],train_df['Pclass'],margins=True).style.background_gradient(cmap='summer_r')
# show chart by Embarked(port)

f, ax = plt.subplots(2, 2, figsize=(20,15))

sns.countplot('Embarked', data=train_df,ax=ax[0,0])

ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Sex',data=train_df,ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived',data=train_df,ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked',hue='Pclass',data=train_df,ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

plt.show()
# check null value

train_df.isnull().sum() # -> 2 'Embarked' value have null value



# predict Embarked null values are 'S'

train_df['Embarked'].fillna('S',inplace=True)



# check null value

train_df.isnull().sum()
# change categorical -> integer

from sklearn import preprocessing



# one-hot-encoding

label_encoder = preprocessing.LabelEncoder()

train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])

train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])
# Use scikit-learn Algorithm model(python)

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics #accuracy measure





# Use 3 data(Pclass, Sex, Embarked) // test data: 30%, train data: 70%

train, test = train_test_split(train_df, test_size=0.3,random_state=0)

target_col = ['Pclass', 'Sex', 'Embarked']



# X: standard(Pclass, Sex, Embarked), Y: Result(Survived)

train_X=train[target_col]

train_Y=train['Survived']

test_X=test[target_col]

test_Y=test['Survived']



features_one = train_X.values

target = train_Y.values



# Use Decision tree model to learn data

tree_model = DecisionTreeClassifier()

tree_model.fit(features_one, target)

dt_prediction = tree_model.predict(test_X)



print('The accuracy of the Decision Tree is',metrics.accuracy_score(dt_prediction, test_Y))

# one-hot-encoding

label_encoder = preprocessing.LabelEncoder()

test_df['Sex'] = label_encoder.fit_transform(test_df['Sex'])

test_df['Embarked'] = label_encoder.fit_transform(test_df['Embarked'])



# predict test data with pre-trained tree model

test_features = test_df[target_col].values

dt_prediction_result = tree_model.predict(test_features)



# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId = np.array(test_df["PassengerId"]).astype(int)

dt_solution = pd.DataFrame(dt_prediction_result, PassengerId, columns = ["Survived"])



# Write your solution to a csv file with the name my_solution.csv

dt_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"]) 