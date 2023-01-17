#I have taken current year of graduation as selection criteria as it was unclear whether it is joining year or passing year and as joining year was not given

#Also for year greater than 2020, I have not assigned any points

#As they are basic things and can be changed easily, I have not tried anything complicated
#Libraries and Functions

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, learning_curve

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import mean_squared_error, accuracy_score

import time

import warnings

warnings.filterwarnings('ignore')
#Understanding the data

df = pd.read_csv('../input/select/Data_Science_2020_v2.csv')

df.head(5)
print(f'This dataset has {df.shape[0]} rows and {df.shape[1]} columns.')
#Data Types

df.dtypes
#Selecting attributes

select = df.copy()

select.drop(['Current City', 'Institute', 'Stream', 'Performance_PG',

                'Performance_UG', 'Performance_12', 'Performance_10'], axis=1, inplace=True)



# Looking for null data

select.isnull().sum()
select_filled = select.fillna('Not defined')

select_filled.isnull().sum()
#Adding score column

select_filled['score'] = 0

select_filled.head(5)
# Define a function to map the values 

def set_value(row_number, assigned_value): 

    return assigned_value[row_number] 



# Create the dictionary 

python_dictionary ={0 : 0, 1 : 3, 2 : 7, 3 : 10} 

  

select_filled['score'] = select_filled['Python (out of 3)'].apply(set_value, args =(python_dictionary, )) 



select_filled.head(5)
#for r

r_dictionary ={0 : 0, 1 : 3, 2 : 7, 3 : 10} 

  

select_filled['score'] = select_filled['score'] + select_filled['R Programming (out of 3)'].apply(set_value, args =(r_dictionary, )) 



select_filled.head(5)
#for data

data_dictionary ={0 : 0, 1 : 3, 2 : 7, 3 : 10} 

  

select_filled['score'] = select_filled['score'] + select_filled['Data Science (out of 3)'].apply(set_value, args =(data_dictionary, )) 



select_filled.head(5)
# substring to be searched 

sub ='Machine Learning'

  

# creating and passsing series to new column 

select_filled["Index"]= select_filled["Other skills"].str.find(sub) 



select_filled['score'] = select_filled['score'] + [0 if x == -1 else 3 for x in select_filled['Index']]



select_filled.head(5)
# substring to be searched 

sub ='Deep Learning'



# creating and passsing series to new column 

select_filled["Index"]= select_filled["Other skills"].str.find(sub) 



select_filled['score'] = select_filled['score'] + [0 if x == -1 else 3 for x in select_filled['Index']]



select_filled.head(5)
# substring to be searched 

sub ='NLP'



# creating and passsing series to new column 

select_filled["Index"]= select_filled["Other skills"].str.find(sub) 



select_filled['score'] = select_filled['score'] + [0 if x == -1 else 3 for x in select_filled['Index']]



select_filled.head(5)
# substring to be searched 

sub ='AWS'



# creating and passsing series to new column 

select_filled["Index"]= select_filled["Other skills"].str.find(sub) 



select_filled['score'] = select_filled['score'] + [0 if x == -1 else 3 for x in select_filled['Index']]



select_filled.head(5)
# substring to be searched 

sub ='SQL'



# creating and passsing series to new column 

select_filled["Index"]= select_filled["Other skills"].str.find(sub) 



select_filled['score'] = select_filled['score'] + [0 if x == -1 else 3 for x in select_filled['Index']]



select_filled.head(5)
# substring to be searched 

sub ='Excel'



# creating and passsing series to new column 

select_filled["Index"]= select_filled["Other skills"].str.find(sub) 



select_filled['score'] = select_filled['score'] + [0 if x == -1 else 3 for x in select_filled['Index']]



select_filled.head(5)
# substring to be searched 

sub ='Data Analytics'



# creating and passsing series to new column 

select_filled["Index"]= select_filled["Other skills"].str.find(sub) 



select_filled['score'] = select_filled['score'] + [0 if x == -1 else 3 for x in select_filled['Index']]



select_filled.head(5)
# substring to be searched 

sub1 ='B.Tech'

sub2 = 'B.E'



# creating and passsing series to new column 

select_filled["Index_1"]= select_filled["Degree"].str.find(sub1) 

select_filled["Index_2"]= select_filled["Degree"].str.find(sub2)



def conditions(s):

    if (s['Index_1'] == -1 ) and (s['Index_2'] == -1):

        return 0

    else:

        return 1



select_filled['Index'] = select_filled.apply(conditions, axis=1)



def conditions(s):

    if (s['Index'] == 1 ) and (s['Current Year Of Graduation'] == 2020 ):

        return 10

    elif (s['Index'] == 1 ) and (s['Current Year Of Graduation'] == 2019 ):

        return 8

    elif (s['Index'] == 1 ) and (s['Current Year Of Graduation'] <= 2018 ):

        return 5

    else :

        return 0



select_filled['score'] = select_filled['score'] + select_filled.apply(conditions, axis=1)

select_filled.head(5)
# substring to be searched 

sub1 ='M.Tech'

sub2 = 'M.Sc'



# creating and passsing series to new column 

select_filled["Index_1"]= select_filled["Degree"].str.find(sub1) 

select_filled["Index_2"]= select_filled["Degree"].str.find(sub2)



def conditions(s):

    if (s['Index_1'] == -1 ) and (s['Index_2'] == -1):

        return 0

    else:

        return 1



select_filled['Index'] = select_filled.apply(conditions, axis=1)



def conditions(s):

    if (s['Index'] == 1 ) and (s['Current Year Of Graduation'] == 2020 ):

        return 7

    elif (s['Index'] == 1 ) and (s['Current Year Of Graduation'] <= 2019 ):

        return 3

    else :

        return 0



select_filled['score'] = select_filled['score'] + select_filled.apply(conditions, axis=1)

select_filled.head(5)
select_filled['result'] = ["Congratulation!" if x >= 40 else "Sorry!" for x in select_filled['score']]

select_filled.head(5)