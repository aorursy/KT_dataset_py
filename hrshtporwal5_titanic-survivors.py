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
import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from collections import Counter

from sklearn import preprocessing 

from keras.utils import to_categorical

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import roc_curve, auc



from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



import warnings

warnings.filterwarnings('ignore')
titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_gender= pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

titanic_test= pd.read_csv('/kaggle/input/titanic/test.csv')

combine = [titanic_train, titanic_test]
## Join train and test datasets to obtain the same number of features during categorical conversion

train_len = len(titanic_train)

titanic_dataset =  pd.concat(objs=[titanic_train, titanic_test], axis=0).reset_index(drop=True)
# Fill empty and NaNs values with NaN

titanic_dataset = titanic_dataset.fillna(np.nan)



# Check for Null values

titanic_dataset.isnull().sum()
#Fill Fare missing values with the median value

titanic_dataset["Fare"] = titanic_dataset["Fare"].fillna(titanic_dataset["Fare"].median())



# As Fare distribution is skewed ,Applying log to Fare to reduce skewness distribution

titanic_dataset["Fare"] = titanic_dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
titanic_train[["Sex","Survived"]].groupby('Sex').mean()
#Fill Embarked nan values of dataset set with 'S' most frequent value

titanic_dataset["Embarked"] = titanic_dataset["Embarked"].fillna("S")
# Filling missing value of Age 



## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(titanic_dataset["Age"][titanic_dataset["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = titanic_dataset["Age"].median()

    age_pred = titanic_dataset["Age"][((titanic_dataset['SibSp'] == titanic_dataset.iloc[i]["SibSp"]) & 

                                       (titanic_dataset['Parch'] == titanic_dataset.iloc[i]["Parch"]) & 

                                       (titanic_dataset['Pclass'] == titanic_dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        titanic_dataset['Age'].iloc[i] = age_pred

    else :

        titanic_dataset['Age'].iloc[i] = age_med
#Binning Age,Fare

titanic_dataset['Age']= pd.qcut(titanic_dataset['Age'], q=10)

titanic_dataset['Fare'] = pd.qcut(titanic_dataset['Fare'], q=13)
# Eature Extraction: Get Title from Name

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in titanic_dataset["Name"]]

titanic_dataset["Title"] = pd.Series(dataset_title)

titanic_dataset["Title"].head()
#Creating Rare & Mrs

titanic_dataset["Title"] = titanic_dataset["Title"].replace(

    ['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 

     'Dona'], 'Rare').replace(['Miss','Ms','Mme','Mlle','Mrs'], 'Mrs')
# Create a family size descriptor from SibSp and Parch

titanic_dataset["Fsize"] = titanic_dataset["SibSp"] + titanic_dataset["Parch"] + 1



# Create new feature of family size

titanic_dataset['Single'] = titanic_dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)

titanic_dataset['FS_S'] = titanic_dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)

titanic_dataset['FS_M'] = titanic_dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

titanic_dataset['FS_L'] = titanic_dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
# Replace the Cabin number by the type of cabin 'X' if not

titanic_dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in titanic_dataset['Cabin'] ])
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 



Ticket = []

for i in list(titanic_dataset.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix

    else:

        Ticket.append("X")

        

titanic_dataset["Ticket"] = Ticket

titanic_dataset["Ticket"].head()
#Label Encoding the Non-Numerical Features

#LabelEncoder basically labels the classes from 0 to n.



non_numeric_features = ['Embarked', 'Sex', 'Cabin', 'Title', 'Age', 'Fare']



label_encoder = preprocessing.LabelEncoder() 

for feature in non_numeric_features:        

    titanic_dataset[feature] = label_encoder.fit_transform(titanic_dataset[feature])
#titanic_dataset1 = pd.DataFrame(to_categorical(titanic_dataset["Pclass"]))

titanic_dataset = pd.get_dummies(titanic_dataset, columns = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title'])
'''cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title']

encoded_features = []



for feature in cat_features:

    encoded_feat = OneHotEncoder().fit_transform(titanic_dataset[feature].values.reshape(-1, 1)).toarray()

    n = titanic_dataset[feature].nunique()

    cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

    encoded_df = pd.DataFrame(encoded_feat, columns=cols)

    encoded_df.index = titanic_dataset.index

    encoded_features.append(encoded_df)

titanic_dataset=pd.concat([titanic_dataset, *encoded_features], axis=1)

'''
# Drop useless variables 

titanic_dataset.drop(labels = ["PassengerId","Ticket","Name","Fsize","SibSp","Parch"], 

                     axis = 1, inplace = True)
titanic_dataset.head()
## Separate train dataset and test dataset



train = titanic_dataset[:train_len]

test = titanic_dataset[train_len:]

test.drop(labels=["Survived"],axis = 1,inplace=True)
## Separate train features and label 

X_train = train.drop(labels = ["Survived"],axis = 1)

Y_train = train['Survived'].values

X_test = test



print('X_train shape: {}'.format(X_train.shape))

print('Y_train shape: {}'.format(Y_train.shape))

print('X_test shape: {}'.format(X_test.shape))
from keras.optimizers import SGD



model = Sequential()

model.add(Dense(27, input_dim=27, kernel_initializer= 'normal' , activation= 'relu' ))

model.add(Dense(14, kernel_initializer= 'normal' , activation= 'relu' ))

model.add(Dense(1, kernel_initializer= 'normal' , activation= 'sigmoid' ))



# Compile model

sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)

model.compile(loss= 'binary_crossentropy' , optimizer='sgd', metrics=['accuracy'])



# Fit the model

model.fit(X_train, Y_train, epochs=150, batch_size=10, verbose=0)
predictions = model.predict(X_test)

predict= pd.DataFrame(predictions)
predict[predict < 0.5] = 0

predict[predict >= 0.5] = 1

predict.columns = ['Survived']
#id=titanic_test['PassengerId'].reset_index(drop=True, inplace=True)

#print(titanic_test['PassengerId'])

output = pd.concat([titanic_test['PassengerId'],predict['Survived'].astype('int') ], axis=1)

output.to_csv('titanic-predictions.csv', index = False)

output.head(100)