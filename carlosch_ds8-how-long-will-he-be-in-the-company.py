import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/HR_comma_sep.csv")

data.head()
#show type of data in dataset

data.info()

print(" rows and columns are: ",data.shape[ 0 ],"-",data.shape[ 1 ] )
#show in each column how many NA is 

data.isnull().sum()
#show data ordinal

salary = data["salary"].unique()

sales = data["sales"].unique()

print("salary are: ",salary )

print( "sales are: ",sales)
#know the max and min element in each column from dataset

maximum = data[ ["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident" ] ].max()

print("Max values in each columns:\n",maximum)

minimum = data[ ["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident" ] ].min()

print("\nMin values in each columns:\n",minimum)
#know the percentage of each sales category

Rpta = data[["sales","salary"]].groupby(["sales"]).size()/14999

print( Rpta.sort_values( ascending = False) )
#know the percentage of each sales and salary  category

Rpta = data[["sales","salary"]].groupby(["sales","salary"]).size()/14999

print( Rpta.sort_values( ascending = False) )
#know the percentage of salary in dataset

Rpta = data[["sales","salary"]].groupby(["salary"]).size()/14999

print( Rpta.sort_values( ascending = False) )
#show type of employee that to be left the company

data_left = data.loc[ data["left"] == 1 ]

Rpta = data_left[["sales","salary"]].groupby(["sales","salary"]).size()

print(Rpta.sort_values(ascending =False ))
#Count the number of accident in work

left = data.groupby(["left"]).size()

print("left people: ",left,"\n")
#state of animus in the company

def Animus( group ):

    if( (group["satisfaction_level"] - group["last_evaluation"] ) < 0  ): return 0

    else:

        return 1

data["animus"] = data.apply(Animus,axis = 1 )

data.groupby( ["animus"] ).size()
#show the features animus, accident and left in data

data.groupby(["left", "Work_accident","animus"]).size()
#analyze the promotion features from dataset

show_promo = data.groupby(["promotion_last_5years"]).size()

#show the featues according from data set

rpta = data.groupby(["left", "Work_accident","promotion_last_5years"]).size()

print( show_promo,"\n",rpta)
#create a the project-time column to mean the time in company

#create a the time invested project 1 column to mean the time per project in all time in company

#create a the time invested project 2 column to mean the time per project in average

data["Time-Project"] = data["average_montly_hours"]/data["time_spend_company"]

data["Time-Invested-Projects_1"] = data["Time-Project"]/data["number_project"]

data["Time-Invested-Projects_1"] = data["average_montly_hours"]/data["number_project"]
Train_data = data