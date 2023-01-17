# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# To begin, I load in the csv-data into a pandas dataframe denoted as'df'

df = pd.read_csv('../input/data.csv')

# To take a peek into our data set

df.head()
list(df.columns)
print(type(df['Age'][0]))

print(type(df['Nationality'][0]))

print(type(df['Overall'][0]))

print(type(df['Potential'][0]))

print(type(df['Value'][0]))

print(type(df['Wage'][0]))
def clean_d(string) :

    last_char = string[-1]

    if last_char == "0":

        return 0

    string = string[1:-1]

    num = float(string)

    if last_char == 'K':

        num = num * 1000

    elif last_char == 'M': 

        num = num * 1000000

    return num
df['Wage_Num'] = df.apply(lambda row: clean_d(row['Wage']), axis=1)

df.head()
df['Value_Num'] = df.apply(lambda row: clean_d(row['Value']), axis=1)

df.head()
df.shape
df.isna().sum()
df = df.dropna(axis=0, subset=['Preferred Foot'])

df.isna().sum()
import seaborn as sns

sns.set()
# See the counts of right-footed players vs. left-footed players

foot_plots = df['Preferred Foot'].value_counts()

foot_plots.plot(kind='bar')
sns.lineplot(x='Overall',y='Wage_Num',data=df)
sns.lineplot(x='Overall',y='Value_Num',data=df)
#compare age with difference in potential to overall

df['Growth_Left'] = df['Potential'] - df['Overall']

sns.lineplot(x='Age',y='Growth_Left',data=df)
sns.lineplot(x='Growth_Left',y='Wage_Num',data=df)
sns.lineplot(x='Age',y='Wage_Num',data=df)

#Observation: line plot might not be best way to visualize this because of outliers
sns.lineplot(x='Age',y='Value_Num',data=df)
sns.lineplot(x='Growth_Left',y='Value_Num',data=df)
sns.lineplot(x='Age',y='Overall',data=df)

#Observation: outliers are significantly influencing overall trend
top_100 = df[:100]

top_100.shape
nationality_100_plots = top_100['Nationality'].value_counts()

nationality_100_plots.plot(kind='bar')

#Observation: European and South American nations dominate in the Top 100 players
age_100_plots = top_100['Age'].value_counts()

age_100_plots.plot(kind='bar')

#Observation: The late 20's are where the majority of Top-100 players are aged
club_100_plots = top_100['Club'].value_counts()

club_100_plots.plot(kind='bar')

#Observation: Almost ALL of the Top-100 players play in Europe, and if not, then in the generous salary-giving Chinese and Japanese nations
#This seaborn function allows you to summarize all basic trends and correlations

sns.pairplot(df, vars=['Age', 'Overall', 'Wage_Num', 'Value_Num', 'Potential', 'Growth_Left'])
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *



# Create target object and call it y

y = df.Overall

# Create X

features = ['Age', 'Value_Num', 'Wage_Num', 'Potential']

X = df[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=1)



# Specify Model

ml_model = DecisionTreeRegressor(random_state=1)

# Fit Model

ml_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = ml_model.predict(val_X)

val_mae = mean_absolute_error(val_y, val_predictions)

print("Validation MAE when using a Decision Tree: {:,.0f}".format(val_mae))

print(train_X)

print(train_y)

print(val_X)

print(val_predictions)

print(val_y)
# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(val_y, rf_val_predictions)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

print(train_X)

print(train_y)

print(val_X)

print(val_predictions)

print(val_y)