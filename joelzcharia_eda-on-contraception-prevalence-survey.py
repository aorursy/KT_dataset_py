#add packages

%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection  import train_test_split
#loading the data

survey=pd.read_csv("../input/contraceptive-prevalence-survey/1987 Indonesia Contraception Prevalence Study.csv", sep=",")

survey.head(5)
#number of rows and columns

print("Rows x Columns:")

print(survey.shape)

#data types of the variables included

print("Feature Type:")

print(survey.dtypes)

#checking for missing values in the columns

print("Missing Values Per Feature:")

print(survey.isnull().sum())
#convert all but age and number of childer to categorical features

survey = survey.astype(str)

survey.Age = survey.Age.astype(int)

survey['Number of Children'] = survey['Number of Children'].astype(int)
#histograms of all the numerical variables

survey.Age.hist(figsize=(10,5),xlabelsize=14, ylabelsize=14)

plt.title("Age Distribution", size=20)
#histograms of all the numerical variables

survey['Number of Children'].hist(figsize=(10,5),xlabelsize=14, ylabelsize=14)

plt.title("Number of Children", size=20)
print(survey.Age.describe())

print(survey['Number of Children'].describe())
#visualizing all the categorical features

#creating list of all columns

columns_bar= survey.select_dtypes(include=object)

columns_bar= list(columns_bar.columns)





#creating a matrix to display all charts

columns_bar_array = np.array(columns_bar)

columns_bar_array = np.reshape(columns_bar_array, (4,2))



#plotting the bar charts

rows = 4 ; columns = 2

f, axes = plt.subplots(rows, columns, figsize=(15,15))

print ("Bar Charts for all Categorical Variables")

for row in range(rows):

    for column in range(columns):

        sns.countplot(survey[columns_bar_array[row][column]], ax = axes[row, column])
# This code turns a text target into numeric to some scikit learn algorithms can process it

le_dep = preprocessing.LabelEncoder()

# to convert into numbers

survey['Contraceptive Method Used'] = le_dep.fit_transform(survey['Contraceptive Method Used'])

survey.dtypes
#setting the target variable as Churn

target_name='Contraceptive Method Used'

target_series=survey[target_name]



#remove the target from current position

survey.drop(columns='Contraceptive Method Used',inplace=True)



#insert the target column in first row

survey.insert(0,target_name,target_series)

survey.head(5)
#getting dummy variables

survey = pd.get_dummies(survey)

sruvey.head(5)
#creating a 70-30 split for trian/test

# split dataset into testing and training

features_train, features_test, target_train, target_test = train_test_split(survey.iloc[:,1:].values, 

                                                                            survey.iloc[:,0].values, test_size=0.30, random_state=0)