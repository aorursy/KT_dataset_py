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



# Write a function which will operate on the data provided and cleanup or combine data as

# needed

def readAndCleanUpData(fileName):    

    df = pd.read_csv(fileName)



    # Drop ticket and cabin. Ticket number is probably not relevant 

    df = df.drop(['Ticket', 'Cabin'], axis = 1)



    # Replace male with 0, female with 1

    df.Sex = df.Sex.map( {'male': 0, 'female': 1} ).astype(int)

    

    # Fill fare with median value

    df['Fare'].fillna( df.Fare.median(), inplace = True)



    # In Embarked we have 3 null values. We will replace this with median values

    # First convert to 0 1 and 2

    df.replace( {'S': 0, 'C':1, 'Q':2}, inplace = True)

    df['Embarked'].fillna( df['Embarked'].median(), inplace = True)

    df.Embarked = df.Embarked.astype(np.int8) # Force it as int



    # Get title of all passengers. This is just for our information to get

    # all relevant titles.

    titles = df.Name.str.split(',').str.get(1).str.split('\.').str.get(0)

    titles.value_counts()

    

    # We have Dr, Sir, Don, Capt, major, Rev. Replace, Jonkheer with Mr    

    df.Name.replace(['Dr', 'Sir', 'Don', 'Capt', 'Major', 'Rev', 'Col', 'Jonkheer'], 'Mr', regex = True, inplace = True)        

    # Replace Ms, Mlle with Miss

    df.Name.replace(['Ms', 'Mlle'], 'Miss', regex = True, inplace = True)    

    # Replace Lady, Countess, Mme with Mrs

    df.Name.replace(['Lady', 'the Countess', 'Mme'], 'Mrs', regex = True, inplace = True)        



    # Now get mean mr, mrs, master ages and replace na with mean

    idx = df['Name'].str.contains('Mr\.')

    val = df.Age[idx].mean()

    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna( val )

    

    idx = df['Name'].str.contains('Mrs\.')

    val = df.Age[idx].mean()

    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna( val )



    idx = df['Name'].str.contains('Miss\.')

    val = df.Age[idx].mean()

    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna( val )



    idx = df['Name'].str.contains('Master\.')

    val = df.Age[idx].mean()

    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna( val )

    

    # Create a new column on relatives

    df['Relatives'] = df['SibSp'] + df['Parch']

    df['iamAlone'] = 0

    df.loc[ df.Relatives == 0, 'iamAlone'] = 1



    # We can delete the name column now    

    pId= df.PassengerId

    df = df.drop(['Name', 'SibSp', 'Parch', 'Relatives', 'PassengerId'], axis = 1)

    return df, pId

    

    



# Read in training data

df, pId = readAndCleanUpData("../input/train.csv")

# Input params

y = df.Survived

X = df.drop(['Survived'], axis = 1) # Remove output from the input

cm = df.corr()





% matplotlib inline

import matplotlib.pylab as plt

# We want to see the correlation matrix

plt.matshow(cm, cmap=plt.cm.gray)

plt.show()



# Print to get an idea of what we get

print( cm.Survived.sort_values(ascending = False))



# All fits

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier()

rnd_clf.fit(X, y)



# looking at accuracy

from sklearn.metrics import accuracy_score

y_pred = rnd_clf.predict(X)

print(rnd_clf.__class__.__name__, accuracy_score(y, y_pred))

    



# Read in test data

t_df, pId = readAndCleanUpData("../input/test.csv")



# Predictions

y_pred = rnd_clf.predict( t_df )



# Create predictions

predictions =  pd.DataFrame( {'PassengerId' : pId,

                             'Survived'    : y_pred} )



# Save the output

predictions.to_csv("my_predictions.csv", index = False)
