# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# get the training and test data as dataframes
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()

test_df.head()
train_df.describe()
# Note that many missing values for Age. Need to do some inputation. I do this in a simple way by basing on the mean for each Sex
# temp1 shows the mean age of Males and Females.
# We then fillna on the missing Age values by defining a fage function and applying it.
temp1 = train_df.groupby('Sex').Age.mean()
def fage(x):
    if x.Sex == 'male':
        return temp1['male']
    if x.Sex == 'female':
        return temp1['female']

train_df.Age.fillna(train_df[train_df.Age.isnull()].apply(fage,axis=1),inplace=True)
train_df.describe()
# I do the same for the test dataset.
temp2 = train_df.groupby('Sex').Age.mean()
def fage(x):
    if x.Sex == 'male':
        return temp2['male']
    if x.Sex == 'female':
        return temp2['female']

test_df.Age.fillna(test_df[test_df.Age.isnull()].apply(fage,axis=1),inplace=True)
test_df.describe()
# Note that test dataset has 1 missing value for 'Fare'. Need to input this using mean fare for that passenger class
test_df[test_df.Fare.isnull()]  # Passenger with missing 'Fare' is in PClass 3

temp3 = test_df.groupby('Pclass').Fare.mean()
temp3
test_df.Fare.fillna(temp3[3],inplace=True)

# Drop unwanted columns
train_df = train_df.drop(['Cabin','Name','Ticket'],axis=1)
test_df = test_df.drop(['Cabin','Name','Ticket'],axis=1)
# OneHotEncoding for Sex and Embarked
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

train_df.head()
#Prepare the datasets for machine learning
y = train_df.Survived
X = train_df.drop(['PassengerId','Survived'],axis=1)
newtest_df = test_df.drop(['PassengerId'],axis=1)

#Using Random Forest, build model on training set and run prediction of survival on test set.
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(X,y)
testpredict = rf.predict(newtest_df)
testpredict
#Outputs the predicted survivals into correct format for submission to Kaggle
predict_df = pd.DataFrame(test_df.PassengerId)
predict_df = predict_df.join(pd.DataFrame(testpredict))
predict_df.columns = ['PassengerId','Survived']
predict_df
predict_df.to_csv("data/output/output.csv")