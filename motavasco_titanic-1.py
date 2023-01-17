# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame, Series

import copy

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def load_and_preprocess(input_file_path):

    df = pd.read_csv(input_file_path)

    df = df.set_index('PassengerId')

    return df
DF_Train_init = load_and_preprocess('../input/train.csv')

DF_Test_init = load_and_preprocess('../input/test.csv')
DF_Train_init.head()
DF_Train_init.describe(include="all")
DF_Train_init.Embarked.value_counts()
DF_Train_init.Pclass.value_counts()
DF_Train_init.Age.plot.box()
DF_Train_init.Age.describe()
print("Missing age values: {}({:.2%}).".format(DF_Train_init.Age.isnull().sum(),DF_Train_init.Age.isnull().sum()/ DF_Train_init.shape[0]))



def extract_title(df):

    FirstName = df["Name"].str.split(',',expand= True)[1]

    df["Title"] = FirstName.str.split('.', expand= True)[0].str.strip()

    return df



def important_title(df):

    df['ImportantTile']= ~((df.Title == "Mr") | (df.Title == "Mrs")| (df.Title == "Miss"))

    #.fillna(True)

    return df
def convert_gender_to_boolean(df):

    # Very XXI century stuff

    df["IsMale"] =  df.Sex.map({"male":True, "female": False})    

    return df
def fill_empy_age_values(df):

    df["Age"] = df.Age.fillna(np.round(df.Age.mean()))

    return df
def drop_unused_variables(df):

    del df["Sex"]

    del df["Name"]

    del df["Ticket"]

    del df["Cabin"]

    del df["Parch"]

    del df["SibSp"]

    return df 
def encode_embarked(df):

    return pd.get_dummies(df, prefix= "Embarked", columns= ["Embarked"])
DF_Train = copy.deepcopy(DF_Train_init)

DF_Train = extract_title(DF_Train)

DF_Train = important_title(DF_Train)

DF_Train = convert_gender_to_boolean(DF_Train)

DF_Train = fill_empy_age_values(DF_Train)

DF_Train = encode_embarked(DF_Train)

DF_Train = drop_unused_variables(DF_Train)
DF_Train.head(20)
DF_Train.Title.value_counts()
DF_Train.corr()