import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



raw_data_train = pd.read_csv('../input/train.csv')

raw_data_test = pd.read_csv('../input/test.csv')

data_combined = pd.concat([raw_data_train, raw_data_test])



data_train.sample(3)
def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    #bins = (-1, 0, 20, 40, 60, 80, 100, 200, 400, 800)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    #group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile', 'A', 'B', 'C', 'D']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df['FareCat'] = categories

    return df



def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(',')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: (x.split(',')[1]).split('.')[0])

    return df    

    

def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    # df = drop_features(df)

    return df



data_combined = transform_features(data_combined)

data_combined.head()
plt.figure(figsize=(10,10))

plot = sns.countplot(y="NamePrefix", data=data_combined)
plt.figure(figsize=(10,10))

sns.barplot(y="NamePrefix", x="Survived", data=data_combined)
plt.figure(figsize=(10,10))

sns.countplot(x="FareCat", hue="Sex", data=data_combined)
plt.figure(figsize=(10,10))

sns.countplot(y="NamePrefix", hue="Sex", data=data_combined)
data_combined.loc[data_combined["NamePrefix"] == " Dona"]