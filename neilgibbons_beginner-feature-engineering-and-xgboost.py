import xgboost as xgb

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/titanic/train.csv")

df.head()
#Dropping these columns for now to quickly set a baseline score

#Will add information from these columns back in over the course of this notebook to improve score

baseline_df = df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
def get_accuracy(dataframe):

    X = dataframe.drop('Survived', axis=1)

    y = dataframe['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = xgb.XGBClassifier(

        n_estimators=100,

        reg_lambda=1,

        gamma=0,

        max_depth=3

    )

    clf.fit(X_train,y_train)

    preds =clf.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(acc)
get_accuracy(baseline_df)
gender_df = baseline_df.join(df['Sex'])

#need to one-hot encode 'Sex' column so that it is in correct format for XGBoost

gender_df['Sex'] = pd.get_dummies(gender_df['Sex'], drop_first =True)

gender_df.head()
#What effect did adding this information have on performance?

get_accuracy(gender_df)
embarked_df = gender_df.join(df['Embarked'])

#need to get information from embarked into a format ready for XGBoost

from sklearn import preprocessing

l_enc = preprocessing.LabelEncoder()

embarked_df['Embarked'] = l_enc.fit_transform(embarked_df['Embarked'].astype(str))

embarked_df.head()
get_accuracy(embarked_df)
def get_title(string):

    titles = ["Capt.","Col.","Don.","Dr.","Major.","Master.","Miss.","Mlle.","Mme.","Mr.","Mrs.","Ms.","Rev."]

    ls = string.split(" ")

    for val in ls:

        if val in titles:

            return val

    return 0

df['Title'] = df['Name'].apply(get_title)

title_df = embarked_df.join(df['Title']) 

title_df.head(3)
#need to label encode the new 'Title' column

from sklearn import preprocessing

enc_title_df = title_df.copy(deep=True)

l_enc = preprocessing.LabelEncoder()

enc_title_df['Title'] = l_enc.fit_transform(enc_title_df['Title'].astype(str))
get_accuracy(enc_title_df)
def rare_title(string):

    if string in ["Capt.","Col.","Don.","Dr.","Major.","Rev."]:

        return 1

    return 0



RareTitle_df= title_df.copy(deep=True)

RareTitle_df['RareTitle'] = RareTitle_df['Title'].apply(rare_title)
#We can see our new column on the far right

RareTitle_df.head(3)
#we need to label encode Title again before putting it into XGBoost

from sklearn import preprocessing

enc_RareTitle_df = RareTitle_df.copy(deep=True)

l_enc = preprocessing.LabelEncoder()

enc_RareTitle_df['Title'] = l_enc.fit_transform(enc_RareTitle_df['Title'].astype(str))
get_accuracy(enc_RareTitle_df)