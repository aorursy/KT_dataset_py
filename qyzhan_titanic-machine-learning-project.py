#import data analysis pacakges

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

# import visual packages: matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Import datasets

train_data = pd.read_csv("../input/train.csv") 

test_data = pd.read_csv("../input/test.csv") 

train_data.head()

train_data.info()

# null values on test_data( age: 177, cabin: 687, embarked: 2)
test_data.info()

# null values on test_data( age: 86, cabin: 327, fare: 1)
# EDA by basic visuals

# Pclass

sns.barplot(x="Pclass", y="Survived", data=train_data)

# Likelihood of survival: 1st class> 2nd> 3rd
# sex

sns.barplot(x="Sex", y="Survived", data=train_data)

# Likelihood of survival: female>> male
# survived by pclass and sex

sns.barplot(x="Sex", y="Survived", hue="Pclass",data=train_data)

# Likelihood of survival

# female: 1st class similar to 2nd class> 3rd; male: 1>2 similar to 3
# Age: string

# num of null values (177) are too large to be ignored, thus, generate random values

avg_age = train_data["Age"].mean()

sd_age = train_data["Age"].std()

rand_age = np.random.randint(avg_age - sd_age, avg_age + sd_age,177)

train_data['Age'][np.isnan(train_data['Age'])] = rand_age

train_data['Age'] = train_data['Age'].astype(int)

train_data.Age.isnull().any()

# no null value of age exists
avg_age1 = test_data["Age"].mean()

sd_age1 = test_data["Age"].std()

rand_age1 = np.random.randint(avg_age1 - sd_age1, avg_age1 + sd_age1,86)

test_data['Age'][np.isnan(test_data['Age'])] = rand_age1

test_data['Age'] = test_data['Age'].astype(int)

test_data.Age.isnull().any()
# Survival by age

train_data['Age range'] = pd.cut(train_data['Age'], 6)

age_data = train_data[['Age range', 'Survived']].groupby(['Age range'], as_index=False).mean()
# number of family member aboard, family size=0 means no family member aboard with the passenger

family_size = train_data['SibSp']+train_data['Parch']

train_data["Family Size"] = family_size

sns.barplot(x="Family Size", y="Survived", data=train_data)

family_size1 = test_data['SibSp']+test_data['Parch']

test_data["Family Size"] = family_size1

# more likely to survive if the family size is between 1 to 3
# check 7 and 10 above

train_data.loc[1:891,:].groupby(['Family Size','Survived']).Name.count()

# % of survival of those passengers with 7 or 10 family members aboard have no chance survive
# Fare and survival

train_data['Fare range'] = pd.cut(train_data['Fare'], 5)

fare_data = train_data[['Fare range', 'Survived']].groupby(['Fare range'], as_index=False).mean()
# Cabin: 687 value missing

# since most of its value are missing, it will not be a good predictor

# Embarked: 2 missing; Ticket

train_data['Embarked'] = train_data['Embarked'].fillna('S')

# the embarked place and ticket choice are not likely to influence the rate of survival

# ignore all 3 features

avg_fare = test_data["Fare"].mean()

test_data['Fare'] = test_data['Fare'].fillna(avg_fare)
# assign numerical variables

train_data['Sex'].replace(['male','female'],[0,1],inplace=True)

train_data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

test_data['Sex'].replace(['male','female'],[0,1],inplace=True)

test_data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
train_data['Age_range']=0

train_data.loc[train_data['Age']<=5,'Age_range']=0

train_data.loc[(train_data['Age']>5)&(train_data['Age']<=10),'Age_range']=1

train_data.loc[(train_data['Age']>10)&(train_data['Age']<=15),'Age_range']=2

train_data.loc[(train_data['Age']>15)&(train_data['Age']<=20),'Age_range']=3

train_data.loc[(train_data['Age']>20)&(train_data['Age']<=30),'Age_range']=4

train_data.loc[(train_data['Age']>30)&(train_data['Age']<=40),'Age_range']=5

train_data.loc[(train_data['Age']>40)&(train_data['Age']<=50),'Age_range']=6

train_data.loc[(train_data['Age']>50)&(train_data['Age']<=60),'Age_range']=7

train_data.loc[train_data['Age']>60,'Age_range']=8



test_data['Age_range']=0

test_data.loc[test_data['Age']<=5,'Age_range']=0

test_data.loc[(test_data['Age']>5)&(test_data['Age']<=10),'Age_range']=1

test_data.loc[(test_data['Age']>10)&(test_data['Age']<=15),'Age_range']=2

test_data.loc[(test_data['Age']>15)&(test_data['Age']<=20),'Age_range']=3

test_data.loc[(test_data['Age']>20)&(test_data['Age']<=30),'Age_range']=4

test_data.loc[(test_data['Age']>30)&(test_data['Age']<=40),'Age_range']=5

test_data.loc[(test_data['Age']>40)&(test_data['Age']<=50),'Age_range']=6

test_data.loc[(test_data['Age']>50)&(test_data['Age']<=60),'Age_range']=7

test_data.loc[test_data['Age']>60,'Age_range']=8
train_data['Fare_range']=0

train_data.loc[train_data['Fare']<=102.466,'Fare_range']=0

train_data.loc[(train_data['Fare']>102.466)&(train_data['Fare']<=204.932),'Fare_range']=1

train_data.loc[(train_data['Fare']>204.932)&(train_data['Fare']<=307.398),'Fare_range']=2

train_data.loc[(train_data['Fare']>307.398)&(train_data['Fare']<=409.863),'Fare_range']=3

train_data.loc[train_data['Fare']>409.863,'Fare_range']=4



test_data['Fare_range']=0

test_data.loc[test_data['Fare']<=102.466,'Fare_range']=0

test_data.loc[(test_data['Fare']>102.466)&(test_data['Fare']<=204.932),'Fare_range']=1

test_data.loc[(test_data['Fare']>204.932)&(test_data['Fare']<=307.398),'Fare_range']=2

test_data.loc[(test_data['Fare']>307.398)&(test_data['Fare']<=409.863),'Fare_range']=3

test_data.loc[test_data['Fare']>409.863,'Fare_range']=4
# drop unneeded features

train_data.drop(['Name','Age','Age range','Ticket','Fare','Fare range','Cabin','PassengerId','SibSp','Parch','Embarked'],axis=1,inplace=True)

test_data.drop(['Name','Age','Ticket','Fare','Cabin','SibSp','Parch','Embarked'],axis=1,inplace=True)
# models

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import GradientBoostingClassifier



train_x = train_data.drop("Survived",axis=1)

train_y = train_data["Survived"]

test_x = test_data.drop("PassengerId",axis=1).copy()
# logistic regression

log_reg = LogisticRegression()

log_reg.fit(train_x, train_y)

y_pred1 = log_reg.predict(test_x)

log_reg.score(train_x, train_y)
# guassian distribution

gaus = GaussianNB()

gaus.fit(train_x, train_y)

y_pred2 = gaus.predict(test_x)

gaus.score(train_x, train_y)
# random forest

rand_forest = RandomForestClassifier()

rand_forest.fit(train_x, train_y)

y_pred3 = rand_forest.predict(test_x)

rand_forest.score(train_x, train_y)
# gradient boosting

grad_boost = GradientBoostingClassifier()

grad_boost.fit(train_x, train_y)

y_pred4 = grad_boost.predict(test_x)

grad_boost.score(train_x, train_y)

# Gradient Boosting has the best prediction 
# submit csv

test_id = test_data['PassengerId']

predict_test = rand_forest.predict(test_x)

submission = pd.DataFrame({ 'PassengerId': test_id, 'Survived': predict_test })

submission.to_csv("submission.csv", index=False)
