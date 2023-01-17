#!/usr/bin/env python3 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Importing the necessary Packages

import os 

import numpy as np 

import pandas as pd 

import tensorflow as tf 

import matplotlib.pyplot as plt 

import seaborn as sns 



# loading the dataset into memory 

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        dataset = os.path.join(dirname, filename)

        

# loading the dataset from the input directory into memroy 

df = pd.read_csv(dataset)

df.fillna(df.mean(), inplace=True)



# Removing columns with NaN values 

df = df.drop(['HTHG', 'HTAG', 'HTR'], axis=1)



# viewing the head of the datast 

df.head() 
# Plotting a graph of the total counts for the potential winnings 

degree_count = df['FTR'].value_counts() 

degree_count.plot(kind='bar')

plt.xlabel('Winnings')

plt.ylabel('Counts')

plt.show() 
# showing the count of the lowest value which would be used for the 

# undersampling process. 

df.loc[df['FTR'] == 'D']['FTR'].count() 
# Balancing the dataset by using undersampling techniques 

shuffled_df = df.sample(frac=2, random_state=4, replace=True)



# getting the value count for the matches that ended in a Draw 

draw = shuffled_df.loc[shuffled_df['FTR'] == 'D'].sample(n=1964, random_state=42)



# getting only 1964 value count for the awayteam winnings

awayteam = shuffled_df.loc[shuffled_df['FTR'] == 'A'].sample(n=1964, random_state=42)



# sampling the Home winnings to 1964 rows 

hometeam = shuffled_df.loc[shuffled_df['FTR'] == 'H'].sample(n=1964, random_state=42)



# normalizing the sampled values into a single column 

df = pd.concat([draw, awayteam, hometeam])



# getting another dataframe for plotting 

raw_df = df
# Plotting a graph of the total counts for the potential winnings 

degree_count = df['FTR'].value_counts() 

degree_count.plot(kind='bar')

plt.xlabel('Winnings')

plt.ylabel('Counts')

plt.show() 
# converting the data and season into a date time format and 

# changing the datetime for the actual data in the matched played. 

time = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# creating the columns for year, month, and day 

df['year'] = time.dt.year 

df['month'] = time.dt.month 

df['day'] = time.dt.day 



# Dropping the data column 

df = df.drop(['Date'], axis=1)

df = df[[

    'year', 'month', 'day', 'HomeTeam', 'AwayTeam', 

    'FTHG', 'FTAG', 'FTR'

]]

# viewing the dataframe 

df.head()
# showing the unique values for all the clubs in total 

df['HomeTeam'].unique() 
# Creating a dictionary to store the numerical values for the respective clubs 

C = {

    "Bayern Munich": 1, "Dortmund": 2, "Duisburg": 3, "FC Koln": 4, 

    "Hamburg": 5, "Leipzig": 6, "M'Gladbach": 7, "Wattenscheid": 8, "Werder Bremen": 9, 

    "Dresden": 10, "Ein Frankfurt": 11, "Freiburg": 12, "Kaiserslautern": 13, 

    "Karlsruhe": 14, "Leverkusen": 15, "Nurnberg": 16, "Schalke 04": 17, 

    "Stuttgart": 18, "Uerdingen": 19, "Bochum": 20, "Munich 1860": 21, 

    "M'gladbach": 22, "Hansa Rostock": 23, "St Pauli": 24, "Dusseldorf": 25, 

    "Bielefeld": 26, "Hertha": 27, "Wolfsburg": 28, "Ulm": 29, 

    "Unterhaching": 30, "Cottbus": 31, "Hannover": 32, "Mainz": 33, 

    "Aachen": 34, "Hoffenheim": 35, "Augsburg": 36, "Greuther Furth": 37, 

    "Fortuna Dusseldorf": 38, "Braunschweig": 39, "Paderborn": 40, "Darmstadt": 41   

}



# mapping the values of the respective dictionary and its corresponding values 

# to the created cleaned dataframe 

df['HomeTeam'] = df['HomeTeam'].map(C)

df['AwayTeam'] = df['AwayTeam'].map(C)



# Creating a dictionary to convert the values in the FTR column into numerical values 

FT = {}

FT['H'] = 1  # Home team won for values on one 

FT['A'] = 2  # Away team won for values of two 

FT['D'] = 3  # the match ended a Draw for values of three. 



# Mapping the values to the FTR column 

df['FTR'] = df['FTR'].map(FT)



# viewing the dataset 

df.head() 
# Getting the numbers of winnings for the HOME TEAM 

hometeam_count = raw_df[raw_df['FTR'] == 'H'].count() 

print(hometeam_count['FTR'], 'winnings for the HOME TEAM')





# Getting the number of winnings for the AWAY TEAM 

awayteam_count = raw_df[raw_df['FTR'] == 'A'].count() 

print(awayteam_count['FTR'], 'winnings for the AWAY TEAM')





# f, axis = plt.subplots(1, 3)

# Plotting a count plot of graph of winnings for the HOME TEAM only

# Against the month inwhich they won the most.

hometeam_count = raw_df[raw_df['FTR'] == 'H']

sns.countplot(x='month', hue='FTR', data=hometeam_count)

plt.grid(True)

plt.ylabel('Count for Home team Winnings')

plt.show()
# Plotting a count plot of winnings for the home team only against 

# the month in which they won the most 

# here the month being the 4th month 

month_4th_count = hometeam_count[hometeam_count['month'] == 4]

ax = sns.countplot(x='day', hue='FTR', data=month_4th_count)

plt.grid(True)

plt.ylabel('Count for Home Team winnings')

plt.show() 



print('Recall that 1 stands for Home Team')
# From the plot below, it show that Dortmund win more home games in April

month_4th_win = month_4th_count['HomeTeam'].value_counts() 

month_4th_win.plot(kind='bar')

plt.xlabel('Winnings')

plt.ylabel('Counts')

plt.show() 
# Sample figsize in inches

fig, ax = plt.subplots(figsize=(20,10))         

# Imbalanced DataFrame Correlation

corr = df.corr()

sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)

ax.set_title("balanced Correlation Matrix", fontsize=14)

plt.show()
# Building a machine learning model to fit the dataset 

# Importing the necessary packages for building the NN model 

from keras.models import Sequential 

from keras.layers import Dense 

from keras.optimizers import SGD 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestRegressor 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import accuracy_score 

from sklearn.svm import SVC 
# saving the dataset into input(x) features and output(y) label 

# and converting into numpy arrays 

X = df.iloc[:, 0:5]

y = df.iloc[:, 7].values



# Splitting the dataset into train test split 

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                   test_size = 0.2, 

                                                   random_state = 20)



# Balancing the X_input features 

X_train.fillna(X_train.mean(), inplace=True)

X_test.fillna(X_test.mean(), inplace=True)
# Displaying the shape of the training dataset 

print('XTrain Shape: {}'.format(X_train.shape))

print('yTrain Shape: {}'.format(y_train.shape))
# Displaying the shape of the testing shape dataset 

print('Xtest Shape: {}'.format(X_test.shape))

print('ytest Shape: {}'.format(y_test.shape))
# Building The Model using Linear Regression 

model = SVC() 

model.fit(X_train, y_train)
# Finding how accurate the model would perform on input data. 

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# To predict, the inputs must be given 

# [year, month, day, HomeTeam, AwayTeam]

new_value = [

    "1995", '3', '4', '22.0', '1.0'

]



# reshaping the new input data so that the model could predict it 

new_value = np.array(new_value).reshape(1, -1)



# predicting the value for full time results 

model.predict(new_value)