#load the necessary libraries

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv',delimiter=",")
df_train.head()
df_test = pd.read_csv('../input/test.csv',delimiter=",")
df_test.head()
def clean_data(df):
    df_elim_cols=df.drop(['Age','Cabin','Ticket','Fare','Embarked'],axis=1) #eliminate columns I don't want
    df_replace_sex = df_elim_cols.replace({'Sex': {'female': 1, 'male': 2}}) #replace Sex data
    return df_replace_sex

clean_train = clean_data(df_train)
clean_test = clean_data(df_test)
clean_train.head()
clean_test.head()
def engineer_title_column(df):
    '''This function will look at the name of each individual, determine the title in the name and assign 
    that title a number. It will then create and append a Title column to the data frame. Note that I am sure
    that there are better ways of doing this!'''
    
    title_list = [] #dummy list of title numbers
    
    for item in df['Name']: #go through each person and identify their title and assign it a number
        if "Mr." in item:
            title_list.append(1)
        elif "Mrs." in item:
            title_list.append(2)
        elif "Miss." in item:
            title_list.append(3)
        elif "Master." in item:
            title_list.append(4)
        elif "Rev." in item:
            title_list.append(5)
        else:
            title_list.append(6) #a "catch-all" for any other title
        
    titles_to_column = pd.Series(title_list) #create a pandas series
    df['Title'] = titles_to_column.values #append the above series to the data frame
    name_dropped = df.drop(['Name'],axis=1) #remove the name column
    return name_dropped
training_df=engineer_title_column(clean_train)
test_df = engineer_title_column(clean_test)
training_df.head()
test_df.head()
X_train = training_df.iloc[:,2:].values #.values turns the dataframe into a numpy array (Gets rid of index)
y_train = training_df.iloc[:,1].values

X_test = test_df.iloc[:,1:].values #.values turns the dataframe into a numpy array (Gets rid of index)
estimator = RandomForestClassifier(n_estimators=10)
rnd_clf = estimator
rnd_clf.fit(X_train,y_train)
y_pred = rnd_clf.predict(X_train)
print(accuracy_score(y_train,y_pred))
confusion_matrix(y_train, y_pred)
test_pred = rnd_clf.predict(X_test)
test_df['Survived'] = test_pred
output_df = test_df.drop(['Pclass','Sex','SibSp','Parch','Title'],axis=1) 
output_df.head()
#finally, export results to a submission file
#output_df.to_csv('submission_file.csv',index=False)
