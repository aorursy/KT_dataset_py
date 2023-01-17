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
#reading in train and test data



train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()

train_data.shape

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
import matplotlib.pyplot as plt

import seaborn as sns
#creating a new column: alone/with family based on Parch and SibSp



train_data["Family"]= ""



train_data.loc[(train_data["SibSp"]|train_data["Parch"])>=1,"Family"]= 1

train_data.loc[train_data["Family"]!= 1, "Family"]=0

print(train_data.head(3))
#creating family for test data



test_data["Family"]= ""



test_data.loc[(test_data["SibSp"]|test_data["Parch"])>=1,"Family"]= 1

test_data.loc[test_data["Family"]!= 1, "Family"]=0

print(test_data.head(3))
#checking for presence of missing values



train_data.info()

#Age, Embarked and Cabin has missing values
#data exploration

#survival rate by Embarking point

sns.catplot(x= 'Embarked',y='Survived', data = train_data, kind = 'bar')



#Cherbourgh has the highest survival rate, while Southampton has the lowest
#Finding out survival rate by gender



sns.catplot(x='Sex', y= 'Survived', data = train_data, kind = 'bar')



#females have much higher survival rate
#Finding out survival percentage by Class



sns.catplot(x='Pclass', y= 'Survived', data = train_data, kind = 'bar')



#class 1 passengers survived the most

#Finding out survival percentage for passengers travelling or alone ( 1 - with family, 0 - alone)



sns.catplot(x='Family', y= 'Survived', data = train_data, kind = 'bar')



#Passengers travelling with family had a higher survival rate than passengers travelling alone
#Survival rate by age



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

female = train_data[train_data['Sex']=='female']

male = train_data[train_data['Sex']=='male']

ax = sns.distplot(female[female['Survived']==1].Age.dropna(), bins=18, label = 'survived', ax = axes[0], kde =False, color="blue")

ax = sns.distplot(female[female['Survived']==0].Age.dropna(), bins=40, label = 'not_survived', ax = axes[0], kde =False, color="red")

ax.legend()

ax.set_title('Female')



ax = sns.distplot(male[male['Survived']==1].Age.dropna(), bins=18, label = 'survived', ax = axes[1], kde = False, color="green")

ax = sns.distplot(male[male['Survived']==0].Age.dropna(), bins=40, label = 'not_survived', ax = axes[1], kde = False, color="red")

ax.legend()

_ = ax.set_title('Male');





#in both males and females, age group between 15-40 has a higher survival rate approx.
#Handling missing values

data = [train_data, test_data]



for dataset in data:

    mean = train_data["Age"].mean()

    std = test_data["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_data["Age"].astype(int)

isnull1 = train_data['Age'].isnull().sum()

isnull1



isnull2 = test_data['Age'].isnull().sum()

isnull2
#filling values for 'Embarked' as 'S' since it is most common



for dataset in data:

    dataset["Embarked"] = dataset["Embarked"].fillna(value='S')

isnull = test_data["Embarked"].isnull().sum()

isnull
#preparing the train and test data ; selecting chosen few columns



features = ["Pclass", "Sex", "Embarked", "Age", "Family"]

X_train = train_data[features]

y_train = train_data['Survived']

X_train.head()

#y_train.head()

X_test = test_data [features]
#encoding and scaling the data



X_train_coded = pd.get_dummies(X_train)

X_test_coded = pd.get_dummies(X_test)





from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_coded)

X_test_scaled = scaler.transform(X_test_coded)

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()



logreg.fit(X_train_scaled, y_train)
#Checking train  accuracy

from sklearn.metrics import accuracy_score

pred = logreg.predict(X_train_scaled)

score = accuracy_score(y_train, pred)

print(score)
#predicting test data



survived = logreg.predict(X_test_scaled)
#saving the output



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': survived})

output.to_csv('my_submission.csv', index=False)





#Hope this notebook was useful, upvote if you found it so

#Cheers !!