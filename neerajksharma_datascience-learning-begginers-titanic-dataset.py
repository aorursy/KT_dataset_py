import numpy as np

import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

    

#read the data with all default parameters

train_df = pd.read_csv("/kaggle/input/titanic/train.csv", index_col = 'PassengerId')   

test_df = pd.read_csv("/kaggle/input/titanic/test.csv", index_col = 'PassengerId')

    

test_df['Survived'] = -888

df = pd.concat((train_df, test_df), axis=0, sort=True) 
# extract rows with Embarked as Null

df[df.Embarked.isnull()]
# how may people embarked at differnt placess and find out the most common embarkment point

df.Embarked.value_counts()



# we can fill the missing values with the S as most people embarked from S.
# we can analyze further and see both of the passenger survived the disaster. So we should find out from which

#Embarked point most people survived that will be more logical. 

pd.crosstab(df[df.Survived!=-888].Survived, df[df.Survived!=-888].Embarked)
#option 2 : explore the fare of each class of each embarkment point 

df.groupby(['Pclass', 'Embarked']).Fare.median()
#see the row where fare is missing. 

df[df.Fare.isnull()]
df.groupby(['Embarked', 'Pclass']).Fare.median()
#calculate the median fare for Embarked in S and Pclass is 3.

median_fare = df.loc[(df.Embarked == 'S') & (df.Pclass == 3),'Fare'].median()
median_fare
df.Name
# Function to extract the title from the name. There is pattern in the titel. Starts with last name, title. first name middle name. 

def getTitle(name):

    first_name_with_title = name.split(',')[1]

    title = first_name_with_title.split('.')[0]

    title = title.strip().lower()  # strip out all of the whitespaces and coverting the title to lower case

    return title
#Binning - qcut() performs quantile based binning. We are splitting the Fare in 4 bins here, Where each bins contains almost equal number of observations.

pd.qcut(df.Fare, 4)
#Specify the name of each bins

pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high'])  # discretization
# lets see the number of obervations in each bins

pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']).value_counts().plot(kind='bar', color='c', rot=0);
def read_data():



    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))

    

    #read the data with all default parameters

    train_df = pd.read_csv("/kaggle/input/titanic/train.csv", index_col = 'PassengerId')   

    test_df = pd.read_csv("/kaggle/input/titanic/test.csv", index_col = 'PassengerId')

    

    test_df['Survived'] = -888

    df = pd.concat((train_df, test_df), axis=0, sort=True) 

    return df
# use map function to apply the function on each Name value row i

df.Name.map(lambda x : getTitle(x))   #alternatively we can use : df.Name.map(getTitle)
# find out how many unique title we have

df.Name.map(lambda x : getTitle(x)).unique()
# We will modify our getTitle function here and we will introduce the dictionary to create custome tile.

# this is to club some of the title

def getTitle(name):

    title_group = {'mr' : 'Mr',

                'mrs' : 'Mrs',

                'miss' : 'Miss',

                'master' : 'Master',

                'don' : 'Sir',

                'rev' : 'Sir', 

                'dr' : 'Officer',

                'mme' : 'Mrs',

                'ms' : 'Mrs',

                'major' : 'Officer',

                'lady' : 'Lady',

                'sir' : 'Sir', 

                'mlle' : 'Miss',

                'col' : 'Officer',

                'capt' : 'Officer',

                'the countess' : 'Lady',

                'jonkheer' : 'Sir', 

                'dona' : 'Lady'

    }

    first_name_with_title = name.split(',')[1]

    title = first_name_with_title.split('.')[0]

    title = title.strip().lower()

    return title_group[title]
# Create title feature

df['Title'] = df.Name.map(lambda x : getTitle(x))
df.head()
#Box plot of Age with title

df[df.Age.notnull()].boxplot('Age', 'Title');
def fill_missing_values(df):

    # Embarked

    df.Embarked.fillna('C', inplace=True)

    # Fare

    median_fare = df.loc[(df.Embarked == 'S') & (df.Pclass == 3),'Fare'].median()

    df.Fare.fillna(median_fare, inplace=True)        

    # age

    title_age_median = df.groupby('Title').Age.transform('median')

    df.Age.fillna(title_age_median, inplace=True)   

    return df
def get_deck(cabin):

    return np.where(pd.notnull(cabin), str(cabin)[0].upper(),'z')      
def reorder_columns(df):

    columns = [column for column in df.columns if column != 'Survived']

    columns = ['Survived'] + columns

    df = df[columns]

    return df
def process_data(df):

    #using the method chaining concept

    return (df

         # create title attribute - then add this

         .assign(Title = lambda x: x.Name.map(get_title))

         # working missing values - start with this

         .pipe(fill_missing_values)   

         # Create Fare_Bin Feature

         .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']))

         # Create ageState

         .assign(AgeState = lambda x : np.where(x.Age >= 18, 'Adult', 'Child'))

         # Creat FamilySize

         .assign(FamilySize = lambda x : x.Parch + x.SibSp + 1)

         # Create IsMother   

         .assign(IsMother = lambda x : np.where(((x.Sex == 'female') & (x.Parch > 0) & (x.Age > 18) & (x.Title != 'Miss')), 1, 0))  

         # Create Deck feature

         .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin))

         .assign(Deck = lambda x : x.Cabin.map(get_deck))   

         # Feature Encoding

         .assign(IsMale = lambda x : np.where(x.Sex == 'male', 1,0))

         .pipe(pd.get_dummies, columns=['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])

         # Add code to drop unnecessarey columns

         .drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis = 1)   

         # Reorder columns

         .pipe(reorder_columns)

           )
    df = read_data()
    df = process_data(df)
#train data

train_df = df.loc[df.Survived != -888]
#test data

columns = [column for column in df.columns if column != 'Survived']

test_df = df.loc[df.Survived == -888, columns]
train_df.head()
test_df.head()
import pandas as pd

import numpy as np

import os
X = train_df.loc[:,'Age':].as_matrix().astype('float')

y = train_df['Survived'].ravel()
print(X.shape, y.shape)
# train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
# Average survival in train and test 

print(f"Mean Survival in train : {np.mean(y_train)}")

print(f"Mean Survival in test : {np.mean(y_test)}")  
from sklearn.dummy import DummyClassifier

# performance matrics

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
# create Model

model_dummy = DummyClassifier(strategy='most_frequent', random_state=0)



# Most frequest will output the majority class. 
# train model

model_dummy.fit(X_train, y_train)
print(f"Score for baseline Model: {model_dummy.score(X_test, y_test)}")

print(f"Accurancy for baseline model: {accuracy_score(y_test, model_dummy.predict(X_test))}")

print(f"Confusion Matrix for baseline model: {confusion_matrix(y_test, model_dummy.predict(X_test))}")

print(f"Precision for baseline model: {precision_score(y_test, model_dummy.predict(X_test))}")

print(f"Recall for baseline model: {recall_score(y_test, model_dummy.predict(X_test))}")
#import function

from sklearn.linear_model import LogisticRegression
# Create model

model_lr_1 = LogisticRegression(random_state=0)

# train model

model_lr_1.fit(X_train, y_train)
print(f"Score for LogisticRegression Model: {model_lr_1.score(X_test, y_test)}")

print(f"Accurancy for LogisticRegression model: {accuracy_score(y_test, model_lr_1.predict(X_test))}")

print(f"Confusion Matrix for LogisticRegression model: {confusion_matrix(y_test, model_lr_1.predict(X_test))}")

print(f"Precision for LogisticRegression model: {precision_score(y_test, model_lr_1.predict(X_test))}")

print(f"Recall for LogisticRegression model: {recall_score(y_test, model_lr_1.predict(X_test))}")