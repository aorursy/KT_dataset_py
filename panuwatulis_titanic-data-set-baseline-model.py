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
#Import Additional Libraries

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

#Loading data set from kaggle 

train_path = "/kaggle/input/titanic/train.csv"

test_path ="/kaggle/input/titanic/test.csv"



train = pd.read_csv(train_path)

test = pd.read_csv(test_path)

#Check training data 

train.head()
# How many rows and columns do we have including train and test data

train.shape, test.shape
#Check imbalance data in Survived Feature

dead_ratio = train['Survived'].value_counts()[0]/len(train)*100

survived_ratio = train['Survived'].value_counts()[1]/len(train)*100



print(f'Dead:{dead_ratio:.2f} %'+' and '+f'Survived:{survived_ratio:.2f} %')
train.info()
#To count value types

train.dtypes.value_counts()
#Check missing value in train set

train.isnull().sum()
#Check missing value in test set

test.isnull().sum()
#How many percentage missing value including train & test set

train_age = train['Age'].isnull().sum()

test_age = test['Age'].isnull().sum()

total_miss_age = train_age+test_age

total_data = len(train)+len(test)



print(f'Age missing value: {total_miss_age/total_data*100:.2f}%')

#Check column name and drop unnecessary

train.columns
train_new = train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

test_new = test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
#Descriptive Statistic for train set

train_new.describe()
#Check Correlation only train data set

corr = train_new.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr, annot=True, cmap='viridis',center=0,fmt='.2f')

plt.title('Correlation Matrix', fontsize=20)

plt.show()

#Create Different plot to check relationship between variables

sns.countplot(x='Survived', data=train)
#Survived by Sex Category

sns.countplot(x='Survived', hue='Sex', data = train)
#Percentage of Dead Male & Female in Each Sex 

#Total male and female in training data

total_male = train['Sex'].value_counts()[0]

total_female = train['Sex'].value_counts()[1]



#Dead male and female in training data

dead_male = train['Sex'][(train['Sex']=='male')&(train['Survived']==0)].count()

dead_female = train['Sex'][(train['Sex']=='female')&(train['Survived']==0)].count()



#Calc 

male_percentage = dead_male/total_male*100

female_percentage = dead_female/total_male*100



print(f"%Dead_Male per Total Male: {male_percentage:.2f} %")

print(f"%Dead_Female per Total Female: {female_percentage:.2f} %")
#Survived by PClass

sns.countplot(x='Survived', hue = 'Pclass', data=train)
#Count PClass from all data set

sns.countplot(x='Pclass', data=train)

pclass_count = train['Pclass'].value_counts()

print('Class 1:', pclass_count[1])

print('Class 2:', pclass_count[2])

print('Class 3:', pclass_count[3])
#Create Age group to see corraltion between Age group and Survived

train_new.loc[(train_new['Age'] < 1),  'AgeGroup'] = 'Baby'

train_new.loc[(train_new['Age'] >= 1) & (train_new['Age']<3),  'AgeGroup'] = 'Child'

train_new.loc[(train_new['Age'] >= 3) & (train_new['Age']<13),  'AgeGroup'] = 'Child'

train_new.loc[(train_new['Age'] >= 13) & (train_new['Age']<20),  'AgeGroup'] = 'Teen'

train_new.loc[(train_new['Age'] >= 20) & (train_new['Age']<=60),  'AgeGroup'] = 'Adult'

train_new.loc[(train_new['Age'] > 60),  'AgeGroup'] = 'Senior'



#Check train data set

train_new.head()
plt.figure(figsize=(10,8))

sns.countplot(x='Survived' ,hue='AgeGroup', data=train_new)
#To see distribution of passenger age

train['Age'].plot.hist()

plt.xlabel('Age')
train['SibSp'].plot.hist()

plt.xlabel('SibSp')
train['Parch'].plot.hist()

plt.xlabel('Parch')
train['Fare'].plot.hist()

plt.xlabel('Fare')
train_new.columns
#Drop categorical data

num_features =['Survived','Pclass','Age','SibSp','Parch','Fare']

train_num =train_new[num_features]
#Drop mission value

train_num = train_num.dropna()

train_num.isnull().sum()
#Separate feature and label

X_train_num = train_num.drop('Survived', axis=1) #features

y_train = train_num['Survived'] #label
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



log_clf = LogisticRegression()

cv_score = cross_val_score(log_clf, X_train_num, y_train, cv = 5, scoring = 'f1_macro' )



print(f'Avg.F1_Macro: {cv_score.mean()*100:.2f}%')

print(f'Std of F1_Macro: {cv_score.std()*100:.2f}%')

#Replace missing value with zero

test_zero = test.replace(np.NaN, 0)



#Select Numerical Features

num_features_2 = ['Pclass','Age','SibSp','Parch','Fare']

test_num = test_zero[num_features_2]



#Fit model to train data set

log_clf.fit(X_train_num, y_train)



#Prediction

y_test_pred = log_clf.predict(test_num)
my_submission = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':y_test_pred})

my_submission
#Save to CSV

my_submission.to_csv('gender_submission_baseline.csv', index=False)