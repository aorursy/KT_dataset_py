#! /usr/bin/python

#

#   titanic03.py

#

#                   Aug/28/2020

# --------------------------------------------------------------------------

import sys

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import  train_test_split

from sklearn.metrics import (roc_curve , auc ,accuracy_score)

# --------------------------------------------------------------------------

# [4-2]:

def convert_proc(df):

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['Embarked'] = df['Embarked'].fillna('S')



    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    df['Embarked'] = df['Embarked'].map( {'S': 0 , 'C':1 , 'Q':2}).astype(int)

#

    df = df.drop(['Cabin','Name','PassengerId','Ticket'],axis =1)

#

    return df

# --------------------------------------------------------------------------

# [4]:

def read_train_proc():

    train_df = pd.read_csv("../input/titanic/train.csv", header=0)

#

    train_df = convert_proc(train_df)

#

    return  train_df



# --------------------------------------------------------------------------

# [6]:

def read_test_proc():

    test_df = pd.read_csv("../input/titanic/test.csv", header=0)

    ids = test_df["PassengerId"].values

#

    test_df = convert_proc(test_df)

#

    return ids,test_df

# --------------------------------------------------------------------------

# [10]:

def submit_proc(ids,output):

    file_submit = "titanic_submit.csv"

#

    dft = pd.DataFrame({'PassengerId': ids, 'Survived': output})

    dft.to_csv(file_submit,index=False)

# --------------------------------------------------------------------------

# [8]:

def predict_proc(train_df,test_df):

#

    train_x = train_df.drop('Survived',axis = 1)

    train_y = train_df.Survived

#

    (train_x , test_x , train_y , test_y) = train_test_split(train_x, train_y , test_size = 0.3 , random_state = 0)

#

    clf = RandomForestClassifier(n_estimators = 10,max_depth=5,random_state = 0)

    clf = clf.fit(train_x , train_y)

    pred = clf.predict(test_x)

#    fpr, tpr , thresholds = roc_curve(test_y,pred,pos_label = 1)

#    auc(fpr,tpr)

    score = accuracy_score(pred,test_y)

    sys.stderr.write("score = %f\n" % score)

#

    predictions = clf.predict(test_df)

#

    return predictions

# --------------------------------------------------------------------------

sys.stderr.write("*** 開始 ***\n")



train_df = read_train_proc()



ids,test_df = read_test_proc()

#

predictions = predict_proc(train_df,test_df)



submit_proc(ids,predictions)



sys.stderr.write("*** 終了 ***\n")

# --------------------------------------------------------------------------