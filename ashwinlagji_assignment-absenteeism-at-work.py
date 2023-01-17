# Imports

import os

import random

import copy

import pandas as pd

import numpy as np

from numpy import percentile

import seaborn as sns

import matplotlib.pyplot as plt

from math import sqrt

from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

from sklearn import preprocessing



%matplotlib inline 



print("Setup Complete")
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Path of the file to read

data_filepath = "/kaggle/input/absenteeism-at-work/Absenteeism_at_work.csv"



# Import Data set

employee_data = pd.read_csv(data_filepath,

                            header=0, 

                            sep=',', 

                            na_values=['.', '??', '', ' ', 'NA', 'na', 'Na', 'N/A', 'N/a', 'n/a'],

                            index_col="ID"

                           )



# Print the first 5 rows of the data

employee_data.head()

# employee_data.to_excel("../employee-data-raw.xlsx")

# Exploratory Data Analysis

print('Shape of dataset: {}'.format(employee_data.shape))



# Data Types of all the variables

print('Feature Type: ')

print('{}'.format(employee_data.dtypes))



# Number of Unique values present in each variable

employee_data.nunique()



# categorising the variables in two category " Continuos" and "Categorical"

continuous_attributes = ['Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Transportation expense',

       'Hit target', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']



categorical_attributes = ['Reason for absence','Month of absence','Day of the week',

                     'Seasons','Disciplinary failure', 'Education', 'Social drinker',

                     'Social smoker', 'Pet', 'Son']





# Transformation DataTypes

for key in employee_data.keys():

    if key in categorical_attributes:

        if key == 'Reason for absence':

#             employee_data[key] = employee_data[key].replace(0,np.nan)

            employee_data[key] = employee_data[key].replace(0,employee_data[key].mean())



        elif key == 'Month of absence':

#             employee_data[key] = employee_data[key].replace(0,employee_data[key].mean())

            employee_data[key] = employee_data[key].replace(0,np.nan)



        employee_data[key] = employee_data[key].astype('category')

print('{}'.format(employee_data.dtypes))

# Number of Unique values present in each variable

# employee_data.nunique()



# Make a copy of employee dataframe

df = employee_data.copy()



print(df[df.isnull().any(axis=1)])



#Creating dataframe with number of missing values

null_data_rows = df[df.isnull().any(axis=1)]

null_data_columns = df.columns[df.isnull().any()]

missing_val = pd.DataFrame(df.isnull().sum())



#Reset the index to get row names as columns

missing_val = missing_val.reset_index()



#Rename the columns

missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percntage'})

missing_val



#Calculate percentage

missing_val['Missing_percntage'] = (missing_val['Missing_percntage']/len(df))*100



#Sort the rows according to decreasing missing percentage

missing_val = missing_val.sort_values('Missing_percntage', ascending = False).reset_index(drop = True)



#Save output to csv file

missing_val.to_csv("Missing_percntage.csv", index = False)



# Return the percentage of missing data in the original dataset

def PerOfMissing(d1,d2):# d1--data by droping the NAN value d2--Original data

    percent_of_missing_data = round( 100 - ((len(d1)/len(d2))*100), 2)

    percent_of_missing_data = str(percent_of_missing_data) + '% of data has Missing value'

    return percent_of_missing_data



# droping all the NAN value from the data and saving the data in data_without_NAN

data_without_NAN = employee_data.dropna()

print (PerOfMissing(data_without_NAN,employee_data))



print(null_data_rows[null_data_columns].head())

print(null_data_columns)

missing_val



# # 5 point summary on 'Absenteeism time in hours'

# mean = df['Absenteeism time in hours'].mean()

# # calculate quartiles

# quartiles = percentile(df['Absenteeism time in hours'], [25, 50, 75])

# # calculate min/max

# data_min, data_max = df['Absenteeism time in hours'].min(), df['Absenteeism time in hours'].max()

# # print 5-number summary

# print('Min: %.3f' % data_min)

# print('Q1: %.3f' % quartiles[0])

# print('Mean: %.3f' % mean)

# print('Median: %.3f' % quartiles[1])

# print('Q3: %.3f' % quartiles[2])

# print('Max: %.3f' % data_max)



sns.boxplot(data=df[['Body mass index']])

fig=plt.gcf()

fig.set_size_inches(5,5)

# ['Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Transportation expense',

#        'Hit target', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']

sns.boxplot(data=df[['Absenteeism time in hours']])

fig=plt.gcf()

fig.set_size_inches(5,5)
# Check the categorical data 

sns.set_style("whitegrid")

sns.catplot(data=df, x='Reason for absence', kind= 'count',size=4, aspect=4)

sns.catplot(data=df, x='Seasons',y="Absenteeism time in hours", size=2, aspect=2)

sns.catplot(data=df, x='Education', kind= 'count', size=2, aspect=2)

# Disciplinary failure                2

# Education                           4

# Son                                 5

# Social drinker                      2

# Social smoker                       2

# Pet                                 6

# Day of week and seasons



sns.catplot(data=df, x='Education',y="Absenteeism time in hours", hue= 'Disciplinary failure', size=8,aspect=2)
#Check the distribution of numerical data using histogram

# ['Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Transportation expense',

#        'Hit target', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']

plt.hist(data=df, x='Service time', bins='auto', label='Service time')

plt.xlabel('Service time')

plt.title("Service time Distribution")
# Check the distribution of numerical data using histogram

d_f = pd.DataFrame(df[continuous_attributes])

hist = d_f.hist(bins=14, figsize=(17,15))


# plt.figure(figsize=(18,8))

sns.jointplot(x='Absenteeism time in hours',y='Seasons',data=df)

sns.jointplot(x='Absenteeism time in hours',y='Age',data=df)

sns.jointplot(x='Absenteeism time in hours',y='Reason for absence',data=df)

s = sns.FacetGrid(data=df,col='Son')

s.map(plt.hist,'Absenteeism time in hours')
s = sns.FacetGrid(data=df,col='Education')

s.map(plt.hist,'Absenteeism time in hours')
s = sns.FacetGrid(data=df,col='Pet')

s.map(plt.hist,'Absenteeism time in hours')
# checking the corellation between all the attributes

correlation_matrix = df.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True)
# corellation using scatter plots



plt.scatter(employee_data['Body mass index'], employee_data['Weight'], alpha=0.5)

plt.title('Scatter plot Weight V/S Body mass index')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
# corellation using scatter plots



plt.scatter(employee_data['Age'], employee_data['Service time'], alpha=0.5)

plt.title('Scatter plot pythonspot.com')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
# Check for outliers using boxplots

# Replace that with MEAN



for i in continuous_attributes:

    # Getting 75 and 25 percentile of variable "i"

    Q3, Q1 = np.percentile(data_without_NAN[i], [75,25])

    MEAN = data_without_NAN[i].mean()

    

    # Calculating Interquartile range

    IQR = Q3 - Q1

    

    # Calculating upper extream and lower extream

    minimum = Q1 - (IQR*1.5)

    maximum = Q3 + (IQR*1.5)

    

    # Replacing all the outliers value to Mean

    data_without_NAN.loc[data_without_NAN[i]< minimum,i] = MEAN

    data_without_NAN.loc[data_without_NAN[i]> maximum,i] = MEAN

    
# Return MAE, MRSE, R², Adjusted R²

def Print_Analysis(y_true, y_pred):

    print ("Mean Square Error (MSE) of data: ", mean_squared_error(y_true,y_pred))

    print ("RMSE of data: ", sqrt(mean_squared_error(y_true,y_pred)))

    print ("R^2 Score(coefficient of determination) : ", r2_score(y_true,y_pred))

    print ('MAE:',mean_absolute_error(y_true,y_pred))
copy_data_without_NAN = copy.deepcopy(data_without_NAN)
# Linear regression



# Normalize the contineous models

scaling_col = continuous_attributes 

# scaling_col = ['Transportation expense', 'Distance from Residence to Work',

#               'Work load Average/day ', 'Hit target','Body mass index']



for i in scaling_col:

    data_without_NAN[i]=(data_without_NAN[i]-min(data_without_NAN[i]))/(max(data_without_NAN[i])-min(data_without_NAN[i]))



data_without_NAN = data_without_NAN.drop(['Weight', 'Service time'],axis=1)

    

X = data_without_NAN.iloc[:, data_without_NAN.columns != 'Absenteeism time in hours']

y = data_without_NAN['Absenteeism time in hours']





#Splitting data into train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20, random_state = 0)

# Importing libraries for Linear Regression

from sklearn.linear_model import LinearRegression



#Train the model

lr_model = LinearRegression().fit(X_train , y_train)



#Perdict for test cases

lr_predictions = lr_model.predict(X_test)



#Create data frame for actual and predicted values

df_lr = pd.DataFrame({'actual': y_test, 'pred': lr_predictions})

print(df_lr.head())



Print_Analysis(y_test, lr_predictions)



print('Intercept: \n', lr_model.intercept_)

print('Coefficients: \n', lr_model.coef_)

# with statsmodels

# Linear regression using ordinary least squares.(OLS)



X = sm.add_constant(X) # adding a constant

 

model = sm.OLS(y.astype(float),  X.astype(float)).fit()

predictions = model.predict(X) 

print_model = model.summary()

print(print_model)
# using SVM



data_without_NAN = copy.deepcopy(copy_data_without_NAN)

# Normalize the contineous models

scaling_col = continuous_attributes 

# scaling_col = ['Transportation expense', 'Distance from Residence to Work',

#               'Work load Average/day ', 'Hit target','Body mass index']



for i in scaling_col:

    data_without_NAN[i]=(data_without_NAN[i]-min(data_without_NAN[i]))/(max(data_without_NAN[i])-min(data_without_NAN[i]))



# data_without_NAN = data_without_NAN.drop(['Weight', 'Service time'],axis=1)

X = data_without_NAN.iloc[:, data_without_NAN.columns != 'Absenteeism time in hours']

y = data_without_NAN['Absenteeism time in hours']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20, random_state = 0)



lab_enc = preprocessing.LabelEncoder()

y_encoded = lab_enc.fit_transform(y_train)



# Fitting Support Vector Machine (SVM) to the opt Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf')

classifier.fit(X_train, y_encoded)



# predicting the test set result

y_pred = classifier.predict(X_test)



#Create data frame for actual and predicted values

df_lr = pd.DataFrame({'actual': y_test, 'pred': y_pred})

print(df_lr.head())



Print_Analysis(y_test, y_pred)