import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test (1).csv")



data_train.head(10)
sns.barplot(x="Embarked",y="Survived",data=data_train,hue="Pclass")
sns.barplot(x="Pclass",y="Survived",data=data_train,hue="Sex")
#Organising Ages, Cabin class and fares



def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin =  df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



#Extracting name prefixes and last names and creating invidual columns for them

def format_names(df):

    df["Lastname"] = df.Name.apply(lambda x: x.split(' ')[0])

    df["Prefix"] = df.Name.apply(lambda x:x.split(" ")[1])

    return df



#Extracting columns - ticket, names and embarked in the new data frame

def drop_features(df):

    return df.drop(["Ticket","Embarked","Name"],axis =1)



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_names(df)

    df = drop_features(df)

    return df



data_train = transform_features(data_train)

data_test = transform_features(data_test)



data_train.head()





    

    

    


