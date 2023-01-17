# General Packages

import pandas as pd

import numpy as np

# Visualization

import seaborn as sns



get_ipython().magic('matplotlib inline')

#plt.rcParams['figure.figsize'] = (16, 8)


def digestData(raw_df):

    

    #Cleanining, use average values for fill gap in the data:

    df= raw_df.drop(['Ticket', 'Cabin'], axis=1)

    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])

    df['Age'] = df['Age'].fillna(df['Age'].mean())

    

    # Extract features Titles from name 

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    import re

    # Define function to extract titles from passenger names

    def get_title(name):

        title_search = re.search(' ([A-Za-z]+)\.', name)

        # If the title exists, extract and return it.

        if title_search:

            return title_search.group(1)

        return ""

    df['Title'] = df['Name'].apply(get_title)

    # Group all non-common titles into one single grouping "Rare"

    df['Title'] = df['Title'].replace(

        ['Lady', 'Countess','Capt', 'Col','Don',

         'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    df = df.drop('Name', axis=1)



    ## Make data more machine-readable

    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # Title

    df['Title'] = df['Title'].map( {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master':3, 'Rare':4} ).astype(int)

    # Embarked

    df['Embarked'] = df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2, 'Master':3, 'Rare':4} ).astype(int)





    # Normalizing data

    from sklearn import preprocessing

    for col in ['Fare', 'Age']:

        transf = df.Fare.reshape(-1,1)

        scaler = preprocessing.StandardScaler().fit(transf)

        df[col] = scaler.transform(transf)

    

    return df
#load training data

train_df = pd.read_csv("../input/train.csv", index_col='PassengerId')

#Digest data

DG_df = digestData(train_df)



#Prepare the data for use in the model

X = DG_df.drop(["Survived"] , axis=1)

y = DG_df["Survived"]

# I choose the SVM  using this Algorithm map

# http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html



from sklearn import svm



clf = svm.SVC()

# Fit data with SVC and create the model

clf.fit(X, y)   
# Use the model on Test data



test_df = pd.read_csv("../input/test.csv", index_col='PassengerId')

DG_test_df = digestData(test_df)

DG_test_df['Survived'] = clf.predict(DG_test_df)



#pd.to_csv("DG_test_df.csv")

DG_test_df['Survived'].to_csv("DG_test_df.csv")
