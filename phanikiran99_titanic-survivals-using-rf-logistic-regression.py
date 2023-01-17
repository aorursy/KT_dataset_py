# Import Libraries 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn import ensemble

import seaborn as sns

import re

import numpy as np

# Ensures graphs to be displayed in ipynb

%matplotlib inline   

sns.set()
# read both train and testdata into dataframe

titanic_df = pd.read_csv('../input/train.csv',header=0)  # Always use header=0 to read header of csv files

titanic_test_df = pd.read_csv('../input/test.csv',header=0)



# Merge Both files so that we get best average values

merged_df = pd.concat([titanic_df,titanic_test_df])

print('Data Sets read into merged_df')
# we can see that the passenger details have been imported and we have all kind of dataformats available for each data field. 

# Next step is to munging the data 

# lets describe and get the info of the data to do so

titanic_df.info() 

print ('*'* 40)

titanic_test_df.info()

# we can observe that couple of information for age,embarked and cabin are missing. Out of which Embarked and Age seems relevent

print ('*'*40)

merged_df.info()
def clean_df(df):

    """ This Function processes Embarked Fill Missing values with 'S' 

                                Gender Convert to 0 or 1 for female and male

                                Family as ParentChild(Parch) + Sibling Spouse (SibSp)"""

    # we can see that we have maximum 'S' so let fill in with 'S' for those missing 2 values 

    df['Embarked'] = df['Embarked'].fillna('S')

    # as it is hard to work on string data in ML lets convert the 'sex' to 'gender' and have values 0,1 for m and f

    df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)

    # now lets cleanup the parch (parent and children) and siblings 

    df['Family'] = df['Parch'] + df['SibSp'] + 1 # Plus 1 because we need to include the passenger

    df = df.drop(['Parch','SibSp','Sex'],axis=1) # Drop these features as we already got them into other

    

    # assign fare for missing values we are considering only the median fare of passenger class 3

    df['Fare'] = df['Fare'].fillna(cleaned_df[cleaned_df['Pclass']==3]['Fare'].median())

    return df
#  assign to a new varaible before that so the data is not lost

cleaned_df = titanic_df.copy()

# Lets display all datatypes that are not good for machine learning, like string/objects

cleaned_df = clean_df(cleaned_df)  # This Cleans Embarked,Gender and Family 

# cleaned_df.info()



# Clean test data too 

cleaned_test_df = titanic_test_df.copy()

cleaned_test_df = clean_df(cleaned_test_df) # This Cleans Embarked,Gender and Family 



# we are cleaning train and test data just for illustration purposes we just need to clean the Merged data

# Clean Merged Data

cleaned_merged_df = merged_df.copy()

cleaned_merged_df = clean_df(cleaned_merged_df) 
# Now let us try to feature engineer the dataset 

# i.e create new features out of existing features , we also need to fix the age of the passengers which are still missing

# 

# below is intermediate function to return only few titles apart from all , just to reduce simular features 

def compressTitle(title):

    if title in ['Mr','Don','Rev']:

        return 'Mr'

    elif title in ['Mrs','Lady','Mme']:

        return 'Mrs'

    elif title in ['Ms','Mlle','Miss']:

        return 'Miss'

    elif title in ['Master','Junior','Jonkheer']:

        return 'Master'

    else:

        return 'Sir'

def feature_engineer(df):

    """ This function will fix the missing values for age in DF,

        Fill Missing Values for Cabin and create binary features

        Fill Missing values for Embarked and create Binary features

        Try to Analyze the ticket number and get some information out of it"""

    # Work on age , But before that lets figure out the Passenger title from the name

    # Instead we can try to create a new model that predicts the age , using any linear regression model. 

    # But first lets try to split the name and try to find the title of the person. 

    df['Title'] = df['Name'].apply(lambda x: re.findall('\s+([a-zA-z]+)+\.',x)[0])

    # by Applying above regex we will get several titles like Capt,, Sir, etc.. which we may want to group similar ones

    # And lets have only 4 Titles , Master, Sir, Mr, Miss 

    

    df['Title'] = df['Title'].apply(compressTitle)

    # now that we got all the titles for all people we can take the age median for different groups seperately

    # we are doing in this way as it is not good to give the average age for infant and for the elder ones 

    all_titles = df['Title'].unique()   # get all titles that are compressed

    for title in all_titles:

        df.loc[df['Title']==title,'Age'] = df.loc[df['Title']==title,'Age'].fillna(df.loc[df['Title']==title]['Age'].mean())

    # With this the age is also fixed now. 

    

    # Work on Cabin , We can observe that the cabin details are starting with charecters and followed by numerics

    # from the detail here http://www.titanicandco.com/inside.html we can find that the First charecter in each Cabin

    # is nothing but the DeckNumber We have A,B,C,D,E,F,G,L,O,T

    # in our data there are so many missing cabin numbers 

    # we can try to fill those missing cabin numbers if we try to extract the relations between people

    # i.e it is very likely if 2 persons are related will be in same cabin/deck 

    # May be we can get it from persons last name, However this model can some times cause problems 

    # as it is not mandate for people belonging to same last name also belong to same family 

    # Also there are atleast 102 last names in the dataset (train)

    # But just assign the last name or Family name for future purpose

    df['Lastname'] = df['Name'].apply(lambda x: re.findall('([A-Za-z]+)\,\s', x)[0])

    

    # Now lets work on deck values, first create new column deck and fill out all the missing values with U (Unknown)

    df['Cabin'] = df['Cabin'].fillna('U0')

    df['Deck'] = df['Cabin'].apply(lambda x:re.findall('([A-Za-z])+',x)[0])

    

    # As the charecter data wont be of much use, let us create binary features for Deck & Title

    df = pd.concat([df,pd.get_dummies(df['Deck']).rename(columns=lambda x: 'Deck_' + str(x))],axis=1)

    df = pd.concat([df,pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))],axis=1)

    # we also have G and O for decks which we dont have any record in train data but if we get test data then we get extra columns

    # Similary Deck T is available in Test but not in train

    if 'Deck_O' not in df.columns:

        df['Deck_O'] = 0

    if 'Deck_L' not in df.columns:

        df['Deck_L'] = 0

    if 'Deck_T' not in df.columns:

        df['Deck_T'] = 0

        

    # Create binary features for Embarked, we can see that most of Embarked as 'S'  so lets fill missing values with 'S'

    df['Embarked'] = df['Embarked'].fillna('S')

    df = pd.concat([df,pd.get_dummies(df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))],axis=1)

    return df



    
def normalize_data(df):

    """ This function will normalize the data so that the model value will not range to low to large extents"""

    for col in df.columns:

        if df[col].dtypes == object:

            pass

        else:

            if col == 'Survived' or col =='PassengerId':

                df = df

            else:

                df[col] = (df[col] - df[col].mean())/df[col].std()

            # this value is nothing but (original - mean)/Standard Deviation

    return df
# Apply Above functions

final_merged_df = feature_engineer(cleaned_merged_df)

final_merged_df = normalize_data(final_merged_df)
# Assign values for which we have survied to train data and nan to test data for predections

final_df = final_merged_df[final_merged_df['Survived'].notnull()]

final_test_df = final_merged_df[final_merged_df['Survived'].isnull()]



# Drop all columns with Objects

train_df = final_df.drop(final_df.columns[final_df.dtypes == object], axis=1) 

test_df = final_test_df.drop(final_test_df.columns[final_test_df.dtypes == object], axis=1)



# Drop the Survived Column from test , This just contains NaN values

test_df = test_df.drop('Survived', axis=1)



# Sort the columns to make sure the train and test data is in same order 

trainCols = sorted(train_df.columns.tolist())

testCols = sorted(test_df.columns.tolist())

train_df = train_df[trainCols].fillna(0)

test_df = test_df[testCols].fillna(0)
# Random Forests Model

X = train_df.drop('Survived',axis=1)

y = train_df['Survived']

X_test = test_df

rfModel = ensemble.RandomForestClassifier(n_estimators=900, random_state = 1, n_jobs=-1)

rfModel.fit(X,y)

y_pred = rfModel.predict(X_test).astype(int)
rfModel.score(X,y)
submission = pd.DataFrame({

        "PassengerId":X_test['PassengerId'],

        "Survived":y_pred

    })

submission.to_csv('titanic_rf.csv',index=False)
# Logistic Regression

lrModel = linear_model.LogisticRegression(random_state = 25)

lrModel.fit(X,y)

y_pred = lrModel.predict(X_test).astype(int)

lrModel.score(X,y)
submission = pd.DataFrame({

        "PassengerId":X_test['PassengerId'],

        "Survived": y_pred

    })

submission.to_csv('titanic.csv',index=False)