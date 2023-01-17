# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Basics

import pandas as pd

import numpy as np



# visualization

import seaborn as sns

sns.set(color_codes=True)

import matplotlib.pyplot as plt

#%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



#os

from os import listdir

from os.path import isfile, join



# machine learning

# import random as rnd

# import operator

# from sklearn.linear_model import LogisticRegression

# from sklearn.svm import SVC, LinearSVC

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.naive_bayes import GaussianNB

# from sklearn.linear_model import Perceptron

# from sklearn.linear_model import SGDClassifier

# from sklearn.tree import DecisionTreeClassifier



# GLOBALS



encoding_type = 'latin-1'







# Get all of the files that have a csv in the file name

onlyfiles = pd.DataFrame([f for f in listdir("../input/") if isfile(join("../input/", f)) and f.find('.csv')>0 ])



# Print the descriptor for the datasets in the library

print("The .csv datasets included in the library are as follows:")



# Print the file names

for x in onlyfiles[0]:

    print(x)



# Create a dictionary of table names

alltables = {'varname':[],

            'filename':[],

            'dataset':[]}



# Loop through the file names and import them, adding a table name that reflects the file name

for i, onlyfile in enumerate(onlyfiles[0]):

    varname = onlyfile[0:3] + str(i)

    alltables['varname'].append(varname)

    alltables['filename'].append(onlyfile)

    

    #test = pd.read_csv('../input/test.csv', encoding='latin-1')

    exec(varname + " = pd.read_csv('../input/"+ onlyfile +"',  encoding='"+ encoding_type + "')") 

    exec("alltables['dataset'].append(" + varname + ")")

    #print(test.columns)

    #exec("print("+ varname + ".columns)") 





# Print the list of table variable names

alltables = pd.DataFrame(alltables)

print(alltables.filter(['varname','filename']))


# Create a dataset of variable types, where we will store the names and types of variables

# We will use this to loop through and get some distributions to find outliers

vartypes = {'name': [],

           'type': []}



# Create an empty data frame for the numeric and string data

num_data = pd.DataFrame()

str_data = pd.DataFrame()



#get metadata for all tables

for dataset in alltables['dataset']:

    g = dataset.columns.to_series().groupby(dataset.dtypes).groups

    # Loop through the metadata, and append the metadata to the vartypes dictionary

    for k, v in g.items():

        for x in v: 

            vartypes['name'].append(x)

            vartypes['type'].append(str(k))

            

            if str(k) == 'float64':

                newframe = pd.DataFrame({x:dataset[x]})

                num_data = pd.concat([newframe, num_data])

                

            #if str(k) == 'object':

                #newframe = pd.DataFrame({x:dataset[x]})

                #str_data = pd.concat([newframe, str_data])

num_data_desc = num_data.describe(percentiles=[.005, .05, .25, .5, .75, .95, .995])

print(num_data_desc)
plt.hist(num_data['Age'].dropna(), bins=100)
for x in range(len(num_data)):

    plt.hist(num_data.iloc[:,x].dropna()) 

    plt.title = num_data.columns[x]

    plt.show()