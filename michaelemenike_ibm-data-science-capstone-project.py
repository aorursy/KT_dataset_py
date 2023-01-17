#gain access to system parameters

import sys

print("Python System Parameters Imported")
#gain access to data storage

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#gain access to system parameters

import sys

print("Python System Parameters Imported")
#import the pandas package for dataframe interaction and manipulation

import pandas as pd

print("Pandas Imported")
#import the visualization package used throughout this notebook

import matplotlib.pyplot as plt

print("Matplotlib Imported")
#import seaborn visualization package

import seaborn as sns

print("Seaborn Imported")
#import seaborn visualization package

import numpy as np

print("Numpy Imported")
 #pretty printing of dataframes in Jupyter notebook

import IPython

from IPython import display

print("IPython Imported")
#collection of machine learning algorithms

import sklearn

from sklearn import ensemble

from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import resample

from sklearn.cluster import KMeans

from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import silhouette_score

print("SciKit Learn Imported")
# for min_max scaling

from mlxtend.preprocessing import minmax_scaling

print('Min-Max Scaling Imported')
#misc libraries

import random

from datetime import datetime

import warnings

import calendar

warnings.filterwarnings('ignore')

print("Miscellaneous Libraries Imported. ")
#csv dataset is read as a dataframe using pandas .read_csv funtion

df_raw = pd.read_csv ('../input/datacollision/Data-Collisions.csv')
#print number of rows and columns

print ("The dataset has %d rows and %d columns" % (df_raw.shape[0] , df_raw.shape[1]))
#get column names

df_raw.columns
#missing training set values

missing_value_sum = df_raw.isnull().sum()

print(missing_value_sum)
# how many total missing values do we have?

total_cells = np.product(df_raw.shape)

total_missing_values = missing_value_sum.sum()

print('The total number of missing (N/A) values in DataFrame are', total_missing_values)
#percentage of data that is missing

percent_missing = (total_missing_values/total_cells) * 100

print("The percentage of missing (N/A) values in the entire df_raw are {} %".format(round(percent_missing)))
#the percentage of null values in column cell

df_raw_column_null = df_raw.isnull().sum().sort_values(ascending=False)/df_raw.shape[0]*100

print("The percentage of missing (N/A) values in df_raw's columns in descending order are:\n{}".format(df_raw_column_null) )
#eliminate columns with N/A values > 40%

df_raw_1 = df_raw.drop(['PEDROWNOTGRNT','EXCEPTRSNDESC','SPEEDING','INATTENTIONIND','INTKEY','EXCEPTRSNCODE','SDOTCOLNUM'], axis = 1)
df_raw_1.head()
#drop columns that do not directly influence the analysis/modelling in the notebook

dropped_colums = ['INCKEY', 'COLDETKEY', 'REPORTNO','STATUS', 'SEVERITYCODE.1', 'ST_COLDESC','SDOT_COLDESC','SEVERITYDESC','INCDATE']

df_raw_2 =  df_raw_1.drop(dropped_colums, axis = 1)                                                                                                                                                                                                                                                                                                                                                                                                          
#show columns in df_raw_3

df_raw_2.columns
#create variable to store rename dictionary

labels = {'SEVERITYCODE':'Severity', 'X':'Longitude', 'Y':'Latitude', 'OBJECTID':'Index', 'ADDRTYPE':'Address_Type', 'LOCATION':'Location',

       'COLLISIONTYPE':'Collision_Type', 'PERSONCOUNT':'People_Hit', 'PEDCOUNT':'Pedestrians_Hit', 'PEDCYLCOUNT':'Bicycles_Hit', 'VEHCOUNT':'Vehicles_Hit',

       'INCDTTM':'Date/Time', 'JUNCTIONTYPE':'Junction_Type', 'SDOT_COLCODE':'SDOT_Collision_Code', 'UNDERINFL':'Intoxicated',

       'WEATHER':'Weather_Condition', 'ROADCOND':'Road_Condition', 'LIGHTCOND':'Lighting_Condition', 'ST_COLCODE':'State_Collision_Code', 'SEGLANEKEY':'Lane_Segment',

       'CROSSWALKKEY':'Crosswalk', 'HITPARKEDCAR':'Parked_Car_Hit'}
#rename column according to description instead of abbreviation

df_raw_3 = df_raw_2.rename(columns = labels)
#show renamed columns

df_raw_3.columns
#make 'Index' the index of dataframe

df_raw_3.set_index('Index', inplace=True)
#display percentage of N/A values in each columns

df_raw_3.isnull().sum().sort_values(ascending=False)/df_raw_3.shape[0]*100
#drop remaining N/A 

df_raw_3.dropna(inplace=True)
#confirm percentage of N/A values after filling

df_raw_3.isnull().sum().sort_values(ascending=False)/df_raw_3.shape[0]*100
#confirm N/A sum

#drop remaining NA

#change Date/Time format to datetime 



df_raw_3['Date/Time'] = df_raw_3['Date/Time'].apply(pd.to_datetime)
df_raw_3[['Date/Time']].info()
#extract Year, Month, Day, Time

df_raw_3['Date'] = pd.DatetimeIndex(df_raw_3['Date/Time']).date

df_raw_3['Hour'] = pd.DatetimeIndex(df_raw_3['Date/Time']).hour

df_raw_3['Weekday'] = df_raw_3['Date/Time'].dt.day_name() 

df_raw_3['Day'] = pd.DatetimeIndex(df_raw_3['Date/Time']).day

df_raw_3['Month'] = df_raw_3['Date/Time'].dt.month_name() 

df_raw_3['Year'] = pd.DatetimeIndex(df_raw_3['Date/Time']).year
df_raw_3.shape
#create dataframe showing season

def season(Month):

    if (Month == 12 or Month == 1 or Month == 2):

       return "Winter"

    elif(Month == 3 or Month == 4 or Month == 5):

       return "Spring"

    elif(Month == 6 or Month== 7 or Month == 8):

       return "Summer"

    else:

       return "Fall"



df_raw_3['Season'] = df_raw_3['Month'].apply(season)
#create dataframe showing time period

timeofdaygroups = {1: "Morning Rush (06:00 - 10:00)",

                   2: "Day (10:01 - 12:00)",

                   3: "Lunch Rush (12:01 - 14:00)",

                   4: "Afternoon (14:01 - 16:00)",

                   5: "After Work Rush (16:01 - 18:00)",

                   6: "Evening (18-22)",

                   7: "Night (22:00 - 6:00)"}



#create daygroup function

def period_tag(hour):

    if hour >= 6 and hour < 10:

        return "1"

    elif hour >= 10 and hour < 12:

        return "2"

    elif hour >= 12 and hour < 14:

        return "3"

    elif hour >= 14 and hour < 16:

        return "4"

    elif hour >= 16 and hour < 18:

        return "5"

    elif hour >= 18 and hour < 22:

        return "6"

    else:

        return "7"



def period_name(value):

    if value == "1":

        return "Morning Rush (06:00 - 10:00)"

    elif value == "2":

        return "Day (10:01 - 12:00)"

    elif value == "3":

        return "Lunch Rush (12:01 - 14:00)"

    elif value == "4":

        return "Afternoon (14:01 - 16:00)"

    elif value == "5":

        return "After Work Rush (16:01 - 18:00)"

    elif value == "6":

        return "Evening (18:01 - 22:00)"

    else:

        return "Night (22:01 - 6:00)"



a = df_raw_3['Hour'].apply(period_tag)

b = a.apply(period_name) 

df_raw_3['Time_Period'] = b
df_raw_3.head(1)





#group hours into respective period based on working class schedule