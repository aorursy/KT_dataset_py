# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)

print("% of women who survived:", rate_women)

print("% of men who survived:", rate_men)
def prepare_data(data, dropfeatures):

    """Takes a dataframe of raw data and returns ML model features

    """

    

    # Build a model excluding non-predictive features. We exclude Survived since we

    # intend to predict that!

    dropfeatures = data.drop(dropfeatures, axis ='columns')



    # Setting missing age values to -1

    dropfeatures["Age"] = data["Age"].fillna(-1)

    

    # Adding the sqrt of the fare feature

    dropfeatures["sqrt_Fare"] = np.sqrt(data["Fare"])

    dropfeatures["sqrt_Fare"] = data["Fare"].fillna(0)    

    #we one-hot encode and add 'Embarked' and 'Sex' features

    features_1h = onehot(dropfeatures)

    clist=pd.DataFrame(data['Cabin'])

    ctrim = clist.applymap(trim_cabin)

    ctrim_1h = cabin_onehot(ctrim)

    df = pd.concat((features_1h,ctrim_1h), axis=1)

    

    return df
def cabin_onehot(cabin):

    """onehot encode the cabins with decks"""

    ohc=pd.get_dummies(cabin,columns=['Cabin'],prefix="cab_")

    return ohc
def onehot(data):

     d=pd.get_dummies(data, columns=['Embarked','Sex'])

     return d
def trim_cabin(cabin):

     """ see if cabin data has something like A12/A123 in it, if so extract 

     first letter, otherwise mark as NA. Valid cabin letters are A-G. 

     /Fragile/: This will fail if whatever is after this is crap but we know all the 

     titanic data there will ever be and it has nothing like that, & shouldn't be brittle on actual data."""

     valid=re.compile(r"^[a-g]{1}\d{2,3}( |$)", re.IGNORECASE)

     #np.isnull blows up so use pd instead nut no pd.nan so use np.nan

     if pd.isnull(cabin) or not valid.match(cabin):

          return  np.nan

     else:

          return cabin[0]
from sklearn.ensemble import RandomForestClassifier

RFCmodel = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)



train_drop=['PassengerId', 'Survived','Fare', 'Name', 'Ticket','Cabin']

test_drop=['PassengerId', 'Fare', 'Name', 'Ticket','Cabin']

features = prepare_data(train_data, train_drop)



RFCmodel.fit(features, train_data["Survived"])



prediction_data=prepare_data(test_data,test_drop)

predictions=RFCmodel.predict(prediction_data) #prediction for test data based on training data



test_Survived = pd.Series(RFCmodel.predict(prediction_data), name="Survived")

results = pd.concat([test_data,test_Survived],axis=1)

results = results.drop(['Fare','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked'], axis ='columns')



results.to_csv('my_submission_4.csv', index=False)

print("Your submission was successfully saved!")