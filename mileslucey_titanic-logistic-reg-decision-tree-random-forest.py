# Dependencies and setup

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import numpy as np

from scipy import stats

import matplotlib.gridspec as gridspec

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, RandomizedSearchCV, cross_val_score, cross_validate

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

from sklearn.compose import ColumnTransformer

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

import statsmodels.api as sm

from sklearn.tree import plot_tree

import os

%matplotlib inline
# Set maximum rows to a high number

pd.set_option('display.max_rows', 100)
# Load datasets

training_data = pd.read_csv('/kaggle/input/titanic/train.csv',index_col=0)

testing_data = pd.read_csv('/kaggle/input/titanic/test.csv',index_col=False)
# Make copies of the original DataFrames in order to prevent errors

training_data_2 = training_data.copy()

testing_data_2 = testing_data.copy()
# Investigate shape of the training dataset

training_data_2.shape
# Investigate shape of the testing dataset

testing_data_2.shape
# Investigate missing values in the training dataset

total_missing_training = training_data_2.isnull().sum().sort_values (ascending = False)

percent_missing_training = round((training_data_2.isnull().sum().sort_values(ascending = False)/len(training_data_2))*100,2)

pd.concat([total_missing_training, percent_missing_training], axis = 1, keys = ['Total','Percent'])
# Investigate missing values in the testing dataset

total_missing_testing = testing_data_2.isnull().sum().sort_values (ascending = False)

percent_missing_testing = round((testing_data_2.isnull().sum().sort_values(ascending = False)/len(testing_data_2))*100,2)

pd.concat([total_missing_testing, percent_missing_testing], axis = 1, keys = ['Total','Percent'])
# Plot a histogram of fares in the training dataset

testing_data_2.Fare.hist(bins=35, density = True, stacked = True, alpha = 0.6, color='royalblue')

testing_data_2.Fare.plot(kind = 'density', color = 'royalblue')

plt.xlabel('Fare ($)')

plt.xlim(-10,200)

plt.title('Distribution of Fare')

plt.show()
# Look at the line item with the missing Fare value

testing_data_2[testing_data_2['Fare'].isnull()]
# Fill in missing Fare values with the median Fare paid by customers of the same passenger class, sex, and port of embarkation

testing_data_2['Fare'] = testing_data_2.groupby(['Pclass','Sex','Embarked'])['Fare'].transform(lambda x: x.fillna(x.median()))
# Look into what the 'Embarked' null rows look like

training_data_2[training_data_2['Embarked'].isnull()]
# Check if there are any other passengers with the same ticket number in the training dataset

training_data_2[training_data_2['Ticket']=='113572']
# Check if there are any other passengers with the same ticket number in the testing dataset

testing_data_2[testing_data_2['Ticket']=='113572']
# Check if there are any other passengers with the same cabin in the training dataset

training_data_2[training_data_2['Cabin']=='B28']
# Check if there are any other passengers with the same cabin in the testing dataset

testing_data_2[testing_data_2['Cabin']=='B28']
# Look at the percentiles of fare prices paid for all first class women

training_data_2[(training_data_2['Pclass'] == 1) & (training_data_2['Sex'] == 'female')].groupby('Embarked')['Fare'].describe()
# Look into the most common port of embarkation for people on cabin level B

training_data_2[training_data_2['Cabin'].str[0]=='B']['Embarked'].value_counts()
# Look at the most common ticket classes for individuals on cabin level B

training_data_2[training_data_2['Cabin'].str[0]=='B']['Pclass'].value_counts()
# Look at the distribution of sexes of passengers inhabiting cabin level B 

training_data_2[training_data_2['Cabin'].str[0]=='B']['Sex'].value_counts()
# Look at the percentiles of fare prices paid by women inhabiting cabin level B

training_data_2[(training_data_2['Cabin'].str[0]=='B') & (training_data_2['Sex'] == 'female')].groupby('Embarked')['Fare'].describe()
# Replace the missing 'Embarked' datapoints with 'C' for 'Cherbourg'

training_data_2.update(training_data_2['Embarked'].fillna('C'))
# Plot a histogram of ages in the training dataset

training_data_2.Age.hist(bins=15, density = True, stacked = True, alpha = 0.6, color='royalblue')

training_data_2.Age.plot(kind = 'density', color = 'royalblue')

plt.xlabel('Age')

plt.xlim(-10,85)

plt.show()
# CREATE A FIGURE SHOWING THE COUNT OF PASSENGERS WITH AGE VALUES POPULATED BY CLASS, SURVIVORSHIP, AND SEX

# Set up the figure with two subplots

fig, (axis1,axis2,axis3) = plt.subplots(1, 3, figsize=(14,6))

# Create a plot showing the count of observations with upper, middle, and lower classes

sns.countplot(x = 'Pclass', data = training_data_2[training_data_2.Age.notnull()], palette = 'Blues_r', ax = axis1)

axis1.set_xticklabels(['Upper','Middle','Lower'])

axis1.set_xlabel('Ticket Class')

# Create a plot showing the count of passengers who survived and did not survive

sns.countplot(x = 'Survived', data = training_data_2[training_data_2.Age.notnull()], palette = 'Blues_r', ax = axis2)

axis2.set_xticklabels(['No','Yes'])

axis2.set_xlabel('Survived?')

# Create a plot showing the count of observations by sex

sns.countplot(x = 'Sex', data = training_data_2[training_data_2.Age.notnull()], palette = 'Blues_r', ax = axis3)

axis3.set_xticklabels(['Male','Female'])

axis3.set_xlabel('Sex')

# Add title and show the graph

plt.text(-3, 495, 'Count of Passengers With Ages', fontsize = 20)

plt.show()
# CREATE A FIGURE SHOWING THE COUNT OF PASSENGERS WITH AGE VALUES EQUAL TO NULL BY CLASS, SURVIVORSHIP, AND SEX

# Set up the figure with two subplots

fig, (axis1,axis2,axis3) = plt.subplots(1, 3, figsize=(14,6))

# Create a plot showing the count of observations with upper, middle, and lower classes

sns.countplot(x = 'Pclass', data = training_data_2[training_data_2.Age.isnull()], palette = 'Oranges_r', ax = axis1)

axis1.set_xticklabels(['Upper','Middle','Lower'])

axis1.set_xlabel('Ticket Class')

# Create a plot showing the count of passengers who survived and did not survive

sns.countplot(x = 'Survived', data = training_data_2[training_data_2.Age.isnull()], palette = 'Oranges_r', ax = axis2)

axis2.set_xticklabels(['No','Yes'])

axis2.set_xlabel('Survived?')

# Create a plot showing the count of observations by sex

sns.countplot(x = 'Sex', data = training_data_2[training_data_2.Age.isnull()], palette = 'Oranges_r', ax = axis3)

axis3.set_xticklabels(['Male','Female'])

axis3.set_xlabel('Sex')

# Add title and show the graph

plt.text(-3, 140, 'Count of Passengers Without Ages', fontsize = 20)

plt.show()
# Replace missing variables with the median of age for individuals grouped by ticket class, sex, and embarked location

training_data_2['Age'] = training_data_2.groupby(['Pclass','Sex','Embarked'])['Age'].transform(lambda x: x.fillna(x.median()))

testing_data_2['Age'] = testing_data_2.groupby(['Pclass','Sex','Embarked'])['Age'].transform(lambda x: x.fillna(x.median()))
# Plot histogram of ages before nulls were replaced and show this histogram in blue

training_data.Age.hist(bins=15, density = True, stacked = True, alpha = 0.6, color='royalblue')

training_data.Age.plot(kind = 'density', color = 'royalblue', label = 'before')

# Plot histogram of ages after null values were replaced and show the histogram in orange

training_data_2.Age.hist(bins=15, density = True, stacked = True, alpha = 0.5, color='orange')

training_data_2.Age.plot(kind = 'density', color = 'orange', label = 'after')

# Create legend and labels

plt.legend()

plt.xlabel('Age')

plt.xlim(-10,85)

plt.show()
# Set up the figure with two subplots

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))

# Create a plot showing the count of observations with upper, middle, and lower classes

sns.countplot(x = 'Pclass', data = training_data_2, palette = 'Blues_r', ax = axis1)

axis1.set_xticklabels(['Upper','Middle','Lower'])

axis1.set_xlabel('Ticket Class')

# Create a plot showing the proportion of people in each class who survived

sns.barplot('Pclass', 'Survived', data = training_data_2, palette = 'Oranges_r', ax = axis2)

axis2.set_xticklabels(['Upper','Middle','Lower'])

axis2.set_xlabel('Ticket Class')

plt.show()
# Set up the figure with two subplots

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))

# Create a plot showing the count of observations with upper, middle, and lower classes

sns.countplot(x = 'Sex', data = training_data_2, palette = 'Blues_r', ax = axis1)

axis1.set_xticklabels(['Male','Female'])

axis1.set_xlabel('Sex')

# Create a plot showing the proportion of people in each class who survive

sns.barplot('Sex', 'Survived', data = training_data_2, palette = 'Oranges_r', ax = axis2)

axis2.set_xticklabels(['Male','Female'])

axis2.set_xlabel('Sex')

plt.show()
# Set up figure with two subplots

plt.figure(figsize=(15,8))

# Create a kernel density estimation plot showing the ages of passengers who survive the shipwreck and color the plot blue

ax1 = sns.kdeplot(training_data_2['Age'][training_data_2.Survived == 1], color = 'royalblue', shade=True)

# Create a kernel density estimation plot showing the ages of passengers who did not survive the shipwreck and color the plot orange

ax2 = sns.kdeplot(training_data_2['Age'][training_data_2.Survived == 0], color = 'orange', shade=True)

# Add titles and legend

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Age for Surviving Population and Deceased Population (Including Replaced Age Values)')

ax2.set(xlabel = 'Age')

plt.show()
# Set up figure with two subplots

plt.figure(figsize=(15,8))

# Create a kernel density estimation plot showing the ages of passengers who survive the shipwreck and color the plot blue

ax1 = sns.kdeplot(training_data['Age'][(training_data.Age.notnull()) & (training_data.Survived == 1)], color = 'royalblue', shade=True)

# Create a kernel density estimation plot showing the ages of passengers who did not survive the shipwreck and color the plot orange

ax2 = sns.kdeplot(training_data['Age'][(training_data.Age.notnull()) & (training_data.Survived == 0)], color = 'orange', shade=True)

# Add titles and legend

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Age for Surviving Population and Deceased Population (Only Including Passengers Whose Ages Originally Specified in the Data)')

ax2.set(xlabel = 'Age')

plt.xlim(-10,85)

plt.show()
# Look at average survival and count of passengers by age

training_data_2[['Age', 'Survived']].groupby(['Age'], as_index = False).agg(['mean', 'count'])
# Set up the figure with two subplots

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))

# Create a plot showing the count of observations with different numbers of siblings / spouses on board the titanic

sns.countplot(x = 'SibSp', data = training_data_2, palette = 'Blues_r', ax = axis1)

axis1.set_xlabel('Number of Siblings / Spouses On Board Titanic')

# Create a plot showing the proportion of with different numbers of siblings / spouses onboard the titanic who survive

sns.barplot('SibSp', 'Survived', data = training_data_2, palette = 'Oranges_r', ax = axis2)

axis2.set_xlabel('Number of Siblings / Spouses On Board Titanic')

plt.show()
# Set up the figure with two subplots

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))

# Create a plot showing the count of observations with different numbers of parents / children on board the titanic

sns.countplot(x = 'Parch', data = training_data_2, palette = 'Blues_r', ax = axis1)

axis1.set_xlabel('Number of Parents / Children On Board Titanic')

# Create a plot showing the proportion of with different numbers of parents / children onboard the titanic who survive

sns.barplot('Parch', 'Survived', data = training_data_2, palette = 'Oranges_r', ax = axis2)

axis2.set_xlabel('Number of Parents / Children On Board Titanic')

plt.show()
# Create a density plot showing the distribution of fares passengers paid

plt.figure(figsize=(15,8))

ax1 = sns.kdeplot(training_data_2['Fare'], color='grey', shade=True)

plt.title('Density Plot of Fare')

ax1.set(xlabel = 'Fare')

plt.show()
# Create a pivot table showing the counts of passengers per ticket number

training_data_3 = training_data_2.copy()

training_data_3.drop(columns = ['Survived'], inplace = True)

combined_data = training_data_3.append(testing_data_2, sort = True)

ticket_counts = combined_data.pivot_table(index = 'Ticket', values = ['Name'], aggfunc = {'Name':'count'})

ticket_counts = ticket_counts.rename(columns = {'Name': 'TicketCount'})
# Merge the dataset with the pivot table in order to add a column showing the number of passengers assigned to the observation's ticket number

training_data_2 = training_data_2.merge(ticket_counts, left_on='Ticket', right_on = 'Ticket')
# Look at the dataset with the new column added to see if ticket fares correspond to individual passengers or to all passengers in groups with the same ticket number

training_data_2.sort_values(by=['TicketCount'], ascending = False).head(20)
# Drop the new ticket count column -- feature engineering is handled later in this notebook

training_data_2.drop(columns = ['TicketCount'], inplace = True)
# Set up the figure with two subplots

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))

# Create a plot showing the count of observations of passengers leaving from different ports

sns.countplot(x = 'Embarked', data = training_data_2, palette = 'Blues_r', ax = axis1)

axis1.set_xlabel('Port of Embarkation')

axis1.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'])

# Create a plot showing the proportion of with different numbers of passengers leaving from different ports who survive

sns.barplot('Embarked', 'Survived', data = training_data_2, palette = 'Oranges_r', ax = axis2)

axis2.set_xlabel('Port of Embarkation')

axis2.set_xticklabels(['Cherbourg','Queenstown', 'Southampton'])

plt.show()
# Define children as under 16 and adults as older than 16 or older

training_data_2['ChildAdult'] = pd.cut(training_data_2['Age'],[0,16,81],  labels = ['child', 'adult'], right = False)

testing_data_2['ChildAdult'] = pd.cut(testing_data_2['Age'],[0,16,81],  labels = ['child', 'adult'], right = False)
# Set up the figure with two subplots

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))

# Create a plot showing the count of observations with different numbers of parents / children on board the titanic

sns.countplot(x = 'ChildAdult', data = training_data_2, palette = 'Blues_r', ax = axis1)

axis1.set_xlabel('Child or Adult')

# Create a plot showing the proportion of with different numbers of family members onboard the titanic who survive

sns.barplot('ChildAdult', 'Survived', data = training_data_2, palette = 'Oranges_r', ax = axis2)

axis2.set_xlabel('Child or Adult')

plt.show()
# Create family size variable

training_data_2['FamilySize'] = training_data_2['SibSp'] + training_data_2['Parch']

testing_data_2['FamilySize'] = testing_data_2['SibSp'] + testing_data_2['Parch']
# Set up the figure with two subplots

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))

# Create a plot showing the count of observations with different numbers of family members on board the Titanic

sns.countplot(x = 'FamilySize', data = training_data_2, palette = 'Blues_r', ax = axis1)

axis1.set_xlabel('Family Size')

# Create a plot showing the proportion of passengers with different numbers of family members onboard the Titanic who survived

sns.barplot('FamilySize', 'Survived', data = training_data_2, palette = 'Oranges_r', ax = axis2)

axis2.set_xlabel('Family Size')

plt.show()
# Create 'WithFamily' variable and define passengers as being with family if family size is greater than 0 

training_data_2['WithFamily'] = np.where((training_data_2['FamilySize'] > 0), 1, 0)

testing_data_2['WithFamily'] = np.where((testing_data_2['FamilySize'] > 0), 1, 0)
# Set up the figure with two subplots

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))

# Create a plot showing the count of observations with different numbers of parents / children on board the Titanic

sns.countplot(x = 'WithFamily', data = training_data_2, palette = 'Blues_r', ax = axis1)

axis1.set_xlabel('Passengers')

axis1.set_xticklabels(['Travel Alone','Travel with Family'])

# Create a plot showing the proportion of passengers with different numbers of family members onboard the titanic who survived

sns.barplot('WithFamily', 'Survived', data = training_data_2, palette = 'Oranges_r', ax = axis2)

axis2.set_xlabel('Passengers')

axis2.set_xticklabels(['Travel Alone', 'Travel with Family'])

plt.show()
# Create variable IsChild that is equal to 1 when the passenger is <16 years old and 0 otherwise

training_data_2['IsChild'] = np.where(training_data_2['ChildAdult'] == 'child', 1 , 0)

testing_data_2['IsChild'] = np.where(testing_data_2['ChildAdult'] == 'child', 1 , 0)

# Create variable IsFemale that is equal to 1 when the passenger is female and 0 otherwise

training_data_2['IsFemale'] = np.where(training_data_2['Sex'] == 'female', 1 , 0)

testing_data_2['IsFemale'] = np.where(testing_data_2['Sex'] == 'female', 1 , 0)
# Create a pivot table that counts the number of women and number of children with each ticket number; This is to be used as a proxy to capture the number of women and children traveling in each group

training_data_4 = training_data_2.copy()

training_data_4.drop(columns = ['Survived'], inplace = True)

combined_data_2 = training_data_4.append(testing_data_2, sort = True)

ticket_counts_2 = combined_data_2.pivot_table(index = 'Ticket', values = ['Name','IsChild', 'IsFemale'], aggfunc = {'Name':'count', 'IsChild':'sum', 'IsFemale':'sum'})

ticket_counts_2 = ticket_counts_2.rename(columns = {'Name': 'TicketCount', 'IsChild':'NumberOfChildren','IsFemale':'NumberOfFemales'})
# Merge the newly-created pivot tables with the training and testing dataset to access the count of women and children traveling in each group

training_data_2 = training_data_2.merge(ticket_counts_2, left_on='Ticket', right_on = 'Ticket')

testing_data_2 = testing_data_2.merge(ticket_counts_2, left_on='Ticket', right_on = 'Ticket')
# Create dummy variable indicating whether or not a passenger is an adult traveling with a child

training_data_2['TravelWChild'] = np.where((training_data_2['NumberOfChildren']>0) & (training_data_2['ChildAdult'] == 'adult'),1,0)

testing_data_2['TravelWChild'] = np.where((testing_data_2['NumberOfChildren']>0) & (testing_data_2['ChildAdult'] == 'adult'),1,0)

# Create dummy variable indicating whether or not a passenger is a man traveling with a woman

training_data_2['TravelWFemale'] = np.where((training_data_2['NumberOfFemales']>0) & (training_data_2['Sex'] == 'male'),1,0)

testing_data_2['TravelWFemale'] = np.where((testing_data_2['NumberOfFemales']>0) & (testing_data_2['Sex'] == 'male'),1,0)
# Set up the figure with two subplots

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))

# Create a plot showing the count of observations of adults traveling with children and adults traveling without children

sns.countplot(x = 'TravelWChild', data = training_data_2[training_data_2['ChildAdult']=='adult'], palette = 'Blues_r', ax = axis1)

axis1.set_xlabel('Passengers')

axis1.set_xticklabels(['Travel Without Child', 'Travel With Child'])

# Create a plot showing the proportion of adults traveling with children and adults traveling without children who survived

sns.barplot('TravelWChild', 'Survived', data = training_data_2[training_data_2['ChildAdult']=='adult'] , palette = 'Oranges_r', ax = axis2)

axis2.set_xlabel('Passengers')

axis2.set_xticklabels(['Travel Without Child', 'Travel With Child'])

plt.show()
# Set up the figure with two subplots

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))

# Create a plot showing the count of observations of men traveling with a female and men traveling without a female

sns.countplot(x = 'TravelWFemale', data = training_data_2[training_data_2['Sex']=='male'], palette = 'Blues_r', ax = axis1)

axis1.set_xlabel('Passengers')

axis1.set_xticklabels(['Travel Without Female', 'Travel With Female'])

# Create a plot showing the proportion of men traveling with women and men traveling without women who survive

sns.barplot('TravelWFemale', 'Survived', data = training_data_2[training_data_2['Sex']=='male'], palette = 'Oranges_r', ax = axis2)

axis2.set_xlabel('Passengers')

axis2.set_xticklabels(['Travel Without Female', 'Travel With Female'])

plt.show()
# Calculate 'FarePerPassenger' as 'Fare' / 'TicketCount' in order to capture the fare related to each individual observation

training_data_2['FarePerPassenger'] = training_data_2['Fare'] / training_data_2['TicketCount']

testing_data_2['FarePerPassenger'] = testing_data_2['Fare'] / testing_data_2['TicketCount']
# Plot the density plot of fare per passenger

plt.figure(figsize=(15,8))

ax1 = sns.kdeplot(training_data_2['FarePerPassenger'], color='grey', shade=True)

plt.title('Density Plot of Fare per Passenger')

ax1.set(xlabel = 'Fare per Passenger')

plt.show()
# Plot density plots of fares paid per passenger for passengers who survive relative to passengers who die 

plt.figure(figsize=(15,8))

ax1 = sns.kdeplot(training_data_2['FarePerPassenger'][(training_data_2.Survived == 1)], color='royalblue', shade=True)

ax2 = sns.kdeplot(training_data_2['FarePerPassenger'][(training_data_2.Survived == 0)], color='orange', shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Fare per Passenger for Surviving Population and Deceased Population')

ax2.set(xlabel = 'Fare per Passenger')

plt.xlim(-10,80)

plt.show()
# Plot density plots of fares paid per passenger for passengers for upper, middle, and lower class passengers

plt.figure(figsize=(15,8))

ax1 = sns.kdeplot(training_data_2['FarePerPassenger'][(training_data_2.Pclass == 1)], color = 'royalblue')

ax2 = sns.kdeplot(training_data_2['FarePerPassenger'][(training_data_2.Pclass == 2)], color = 'orange')

ax3 = sns.kdeplot(training_data_2['FarePerPassenger'][(training_data_2.Pclass == 3)], color = 'dimgray')

plt.legend(['Upper Class', 'Middle Class', 'Lower Class'])

plt.title('Density Plot of Fare per Passenger for People of Different Classes')

ax2.set(xlabel = 'Fare per Passenger')

plt.xlim(-10,80)

plt.show()
# Plot density plots of fares paid per passenger for passengers who survive relative to passengers who die for upper class passengers only 

plt.figure(figsize=(15,8))

ax1 = sns.kdeplot(training_data_2['FarePerPassenger'][(training_data_2.Survived == 1) & (training_data_2.Pclass == 1)], color='royalblue', shade=True)

ax2 = sns.kdeplot(training_data_2['FarePerPassenger'][(training_data_2.Survived == 0) & (training_data_2.Pclass == 1)], color='orange', shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Fare per Passenger for Surviving Population and Deceased Population of Upper Class Passengers')

ax2.set(xlabel = 'Fare per Passenger')

plt.xlim(-10,80)

plt.show()
# Plot density plots of fares paid per passenger for passengers who survive relative to passengers who die for middle class passengers only

plt.figure(figsize=(15,8))

ax1 = sns.kdeplot(training_data_2['FarePerPassenger'][(training_data_2.Survived == 1) & (training_data_2.Pclass == 2)], color='royalblue', shade=True)

ax2 = sns.kdeplot(training_data_2['FarePerPassenger'][(training_data_2.Survived == 0) & (training_data_2.Pclass == 2)], color='orange', shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Fare per Passenger for Surviving Population and Deceased Population of Middle Class Passengers')

ax2.set(xlabel = 'Fare per Passenger')

plt.xlim(-3,19)

plt.show()
# Plot density plots of fares paid per passenger for passengers who survive relative to passengers who die for lower class passengers only

plt.figure(figsize=(15,8))

ax1 = sns.kdeplot(training_data_2['FarePerPassenger'][(training_data_2.Survived == 1) & (training_data_2.Pclass == 3)], color='royalblue', shade=True)

ax2 = sns.kdeplot(training_data_2['FarePerPassenger'][(training_data_2.Survived == 0) & (training_data_2.Pclass == 3)], color='orange', shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Fare per Passenger for Surviving Population and Deceased Population of Lower Class Passengers')

ax2.set(xlabel = 'Fare per Passenger')

plt.xlim(-1,15)

plt.show()
# Since data has been cleaned and new features produces, create a new dataframe for the cleaned data

training_data_cleaned = training_data_2.copy()

testing_data_cleaned = testing_data_2.copy()
# Create a dataframe for the data used in training the logistic regression model

log_res_training = training_data_cleaned.copy()
# Drop variables that are not being used

log_res_training.drop(columns = ['Cabin','Age','SibSp', 'Parch', 'FamilySize','Name', 'Ticket','Sex', 'ChildAdult','TicketCount','NumberOfChildren', 'NumberOfFemales', 'Fare'], inplace = True)
# Create binary variables for all categorical variables and drop the first category to avoid perfect multicollinearity

log_res_training = pd.get_dummies(log_res_training, columns = ['Embarked'], drop_first = True)
# Apply standard scaler to 'Fare' column

fare_per_passenger = log_res_training[['FarePerPassenger']].values

fare_per_passenger = StandardScaler().fit_transform(fare_per_passenger)

log_res_training['FarePerPassenger'] = fare_per_passenger
# Take a look at the final dataset

log_res_training.head()
# Define the dependent and independent variables

y_logistic = log_res_training.iloc[:,0].values

X_logistic = log_res_training.iloc[:,1:].values
# Fit a logistic model to the data

logistic_regression_classifier = sm.Logit(endog = y_logistic, exog = X_logistic).fit()
# Show coefficients of the logistic regression

logistic_regression_classifier.summary()
# The 'TravelWChild' variable (Binary variable indicating whether or not the individual traveled with a child) has a high p-value revealing that the variable is not statistically significant in this model. Remove the 'TravelWChild' variable

X_logistic = np.delete(X_logistic,4, axis = 1)
# Fit a logistic model to the data (with the 'TravelWChild' variable removed) 

logistic_regression_classifier = sm.Logit(endog = y_logistic, exog = X_logistic).fit()
# Show coefficients of the logistic regression

logistic_regression_classifier.summary()
# The 'Embarked_Q' variable (Binary variable indicating whether or not the individual embarked the Titanic from the Queenstown port) has a high p-value revealing that the variable is not statistically significant in this model. Remove the 'Embarked_Q' variable

X_logistic = np.delete(X_logistic,6, axis = 1)
# Fit a logistic model to the data (with the 'TravelWChild' and 'Embarked_Q' variables removed)

logistic_regression_classifier = sm.Logit(endog = y_logistic, exog = X_logistic).fit()
# Show coefficients of the logistic regression

logistic_regression_classifier.summary()
# The 'Embarked_S' variable (Binary variable indicating whether or not the individual embarked the Titanic from the Southampton port) has a high p-value revealing that the variable is not statistically significant in this model. Remove the 'Embarked_S' variable

X_logistic = np.delete(X_logistic,6, axis = 1)
# Fit a logistic model to the data (with the 'TravelWChild', 'Embarked_Q', and 'Embarked_S' variables removed)

logistic_regression_classifier = sm.Logit(endog = y_logistic, exog = X_logistic).fit()
# Show coefficients of the logistic regression

logistic_regression_classifier.summary()
# The 'WithFamily' variable (binary variable indicating whether or not a passenger is traveling with family) has a high p-value revealing that the variable is not statistically significant in this model. Removing the 'WithFamily' variable

X_logistic = np.delete(X_logistic,1, axis = 1)
# Fit a logistic model to the data (with the 'TravelWChild', 'Embarked_Q', 'Embarked_S', and 'WithFamily' variables removed)

logistic_regression_classifier = sm.Logit(endog = y_logistic, exog = X_logistic).fit()
# Show coefficients of the logistic regression

logistic_regression_classifier.summary()
# The 'TravelWFemale' variable (variable indicating whether a male passenger is traveling with a female) has a high p-value revealing that the variable is not statistically significant in this model. Removing the 'TravelWFemale' variable

X_logistic = np.delete(X_logistic,3, axis = 1)
# Fit a logistic model to the data (with the with the 'TravelWChild', 'Embarked_Q', 'Embarked_S', 'WithFamily', and 'TravelWFemale' variables removed)

logistic_regression_classifier = sm.Logit(endog = y_logistic, exog = X_logistic).fit()
# Show coefficients of the logistic regression

logistic_regression_classifier.summary()
# Fit the final logistic regression classifier

logistic_regression_classifier = LogisticRegression(random_state = 0)

logistic_regression_classifier.fit(X_logistic, y_logistic)
# Make a copy of the cleaned training data to avoid errors

dec_tree_training = training_data_cleaned.copy()
# Drop variables not being used

dec_tree_training.drop(columns = ['Cabin','SibSp', 'Parch','Name', 'Ticket','Sex', 'ChildAdult','IsChild','WithFamily','TicketCount','NumberOfChildren', 'NumberOfFemales', 'Fare'], inplace = True)
# Create dummy variables for nominal categorical variables

dec_tree_training = pd.get_dummies(dec_tree_training, columns = ['Embarked'])
# Look at the dataset

dec_tree_training.head()
# Define x and y variables for the decision tree

y_tree = dec_tree_training.iloc[:,0].values

X_tree = dec_tree_training.iloc[:,1:].values
# Create a decision tree to the data without passing any parameters

tree_1 = DecisionTreeClassifier()

tree_1 = tree_1.fit(X_tree, y_tree)
# Show chart of this initial decision tree

plt.figure(figsize=(75,40))

tree_1_image = plot_tree(tree_1, 

              feature_names=dec_tree_training.iloc[:,1:].columns, 

              class_names={0:'Died',1:'Survived'},

              filled=True, 

              rounded=True, 

              fontsize=14)
# Define a list of max_depths, min_samples_leaves, and min_samples_splits

max_depths = list(range(1,41))

min_samples_leaves = list(range(1,41))

# min_samples_splits = list(range(1,41))
# Pass the parameters into GridSearchCV

grid_decision_tree = GridSearchCV(DecisionTreeClassifier(),{'max_depth': max_depths, 'min_samples_leaf': min_samples_leaves}, cv = 5,scoring = 'roc_auc', n_jobs = -1)
# Use GridSearchCV to figure out the best possible parameters to pass

grid_decision_tree.fit(X_tree, y_tree)
# Fit the newly optimized decision tree

tree_2 = DecisionTreeClassifier(max_depth = grid_decision_tree.best_params_['max_depth'], min_samples_leaf = grid_decision_tree.best_params_['min_samples_leaf'])

tree_2 = tree_2.fit(X_tree, y_tree)
# Display the newly optimized decision tree

plt.figure(figsize=(50,20))

tree_2_image = plot_tree(tree_2, 

              feature_names=dec_tree_training.iloc[:,1:].columns, 

              class_names={0:'Died',1:'Survived'},

              filled=True, 

              rounded=True, 

              fontsize=14)
# Create copy of the training data to use for random forest model training

rndm_frst_training = training_data_cleaned.copy()
# Drop columns that are not being used in the model

rndm_frst_training.drop(columns = ['Cabin','SibSp', 'Parch', 'WithFamily','Name', 'Ticket','Sex', 'ChildAdult','TicketCount','NumberOfChildren', 'NumberOfFemales', 'Fare', 'IsChild'], inplace = True)
# Create dummy variables for the 'Embarked' feature

rndm_frst_training = pd.get_dummies(rndm_frst_training, columns = ['Embarked'])
# Look at the final dataset

rndm_frst_training.head()
# Define x and y variables

y_forest = rndm_frst_training.iloc[:,0].values

X_forest = rndm_frst_training.iloc[:,1:].values
# Define the ranges of parameter values to test

n_estimators = list(range(1,126))

max_depths = list(range(1,34))

min_samples_splits = list(range(1,34))

min_samples_leaves = list(range(1,34))
# Pass the parameters into RandomizedSearchCV

grid_random_forest = RandomizedSearchCV(RandomForestClassifier(),{'n_estimators':n_estimators, 'max_depth': max_depths, 'min_samples_leaf': min_samples_leaves, 'min_samples_split': min_samples_splits}, cv = 5,scoring = 'roc_auc', n_iter = 1000, n_jobs = -1, random_state = 0)
# Use RandomizedSearchCV to figure out the best possible parameters to pass

grid_random_forest.fit(X_forest, y_forest)
# Fit the newly optimized decision tree

forrest_1 = RandomForestClassifier(random_state = 0, max_features = 3, n_estimators = grid_random_forest.best_params_['n_estimators'], max_depth = grid_random_forest.best_params_['max_depth'], min_samples_leaf = grid_random_forest.best_params_['min_samples_leaf'], min_samples_split = grid_random_forest.best_params_['min_samples_split'])

forrest_1 = forrest_1.fit(X_tree, y_tree)
# Measure performance of the logistic regression model using 5 fold cross validation

scores_logistic_regression = cross_validate(logistic_regression_classifier, X_logistic, y_logistic, cv = 5, scoring = ['accuracy','precision', 'recall', 'f1', 'roc_auc'])
# Print the performance scores for the logistic regression

print('Logistic Regression Test Results')

print('--------------------------------------')

print('accuracy: ' + str(scores_logistic_regression['test_accuracy'].mean()))

print('precision score: ' + str(scores_logistic_regression['test_precision'].mean()))

print('recall score: ' + str(scores_logistic_regression['test_recall'].mean()))

print('f1 score: ' + str(scores_logistic_regression['test_f1'].mean()))
# Measure performance of the decision tree model using 5 fold cross validation

scores_decision_tree = cross_validate(tree_2, X_tree, y_tree, cv = 5, scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
# Print the performance scores for the decision tree

print('Decision Tree Test Results')

print('--------------------------------------')

print('accuracy: ' + str(scores_decision_tree['test_accuracy'].mean()))

print('precision score: ' + str(scores_decision_tree['test_precision'].mean()))

print('recall score: ' + str(scores_decision_tree['test_recall'].mean()))

print('f1 score: ' + str(scores_decision_tree['test_f1'].mean()))
# Measure performance of the random forest model using 5 fold cross validation

scores_random_forest = cross_validate(forrest_1, X_forest, y_forest, cv = 5, scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
# Print the performance scores for the random forest

print('Random Forest Test Results')

print('--------------------------------------')

print('accuracy: ' + str(scores_random_forest['test_accuracy'].mean()))

print('precision score: ' + str(scores_random_forest['test_precision'].mean()))

print('recall score: ' + str(scores_random_forest['test_recall'].mean()))

print('f1 score: ' + str(scores_random_forest['test_f1'].mean()))
# Create the submission as its own dataframe

competition_submission = testing_data_cleaned.copy()
# Drop columns that are not being used in the model

competition_submission.drop(columns = ['Cabin','SibSp', 'Parch', 'WithFamily','Name', 'Ticket','Sex', 'ChildAdult','TicketCount','NumberOfChildren', 'NumberOfFemales', 'Fare', 'IsChild'], inplace = True)
# Create dummy variables for the 'Embarked' feature

competition_submission = pd.get_dummies(competition_submission, columns = ['Embarked'])
# Look at the final dataset

competition_submission.head()
# Define the independent variables

X_competition = competition_submission.iloc[:,1:].values
# Create survived column with the predictions included

competition_submission['Survived'] = grid_random_forest.predict(X_competition)
# Drop the columns that are not needed

competition_submission.drop(columns = ['Pclass', 'Age', 'TravelWFemale', 'FamilySize', 'IsFemale', 'TravelWChild', 'FarePerPassenger', 'Embarked_C', 'Embarked_Q', 'Embarked_S'], inplace = True)
# Export the best performing model's predictions

competition_submission.to_csv('/kaggle/working/submission_rf.csv', index = False)