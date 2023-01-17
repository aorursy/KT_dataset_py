# data analysis and wrangling

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np  # linear algebra

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV



#Learning curve

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import validation_curve
'''

import csv



#The Path to the folder containing the data files

PATH = "../input/"



#Read a CSV File and Load it as a List

def readFile(file_name):

    with open(PATH+file_name) as csvfile:

        data = list(csv.reader(csvfile))

    return data



#Convert the Data from a List to a Panda's Data Frame

def ConvertToDF(data):

    df = pd.DataFrame(data[1:], columns = data[0])

    return df

'''
PATH = "../input/"



import csv

from IPython.display import display 



'''The TRAIN data file'''

#train_data = readFile("train.csv") # Load TRAIN data

#train_df = ConvertToDF(train_data) # Convert TRAIN data to DF

train_df = pd.read_csv(PATH+'train.csv')



train_df[:12]
train_df.info()
train_df.describe() #describing basic statistics of numerical values
train_df.describe(include=['O'])
ISchild = train_df.Age.between(0,12)

child12 = train_df[ISchild]

display(child12)

child12.info()
child12_dead = child12[child12.Survived==0]

child12_alive = child12[child12.Survived==1]

print ("Number of children who lived = ", child12_alive.PassengerId.count()

       ,"\nNumber of children who died = ",child12_dead.PassengerId.count()

       ,"\nNumber of all children under 12 = ", child12.shape[0]

       ,"\nPercent of survival among children under the age of 12 equals "

       ,round((child12_alive.shape[0]/child12.shape[0])*100,2), "%")
'''Check for null'''

# Cabin

ISnullCabin = train_df.Cabin.isnull()

# null cabins

nullCabins = train_df[ISnullCabin==True]

nullCabins.describe()
nullCabins_Survived = nullCabins[nullCabins.Survived==1]

display(nullCabins_Survived.head())

nullCabins_Survived.info()
#correlation matrix

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(train_df.corr(), vmax=.8, square=True);