#import libraries for data analysis

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly

import seaborn as sns



#import library to ignore warnings (just for Kaggle)

import warnings

warnings.filterwarnings('ignore')
#read csv data

homicide_reports = '../input/database.csv'

df = pd.read_csv(homicide_reports)



#create a new copy of the data

homicide_df = df.copy()



#drop columns that will not be used for analysis

homicide_df = (homicide_df.drop(['Record ID','Agency Code','Agency Name','Agency Type',

                                 'Victim Ethnicity','Perpetrator Ethnicity','Record Source'],axis=1))
#identify number of rows and columns of the data

print(homicide_df.shape)
#convert datatypes of columns to object

homicide_df[['Incident', 'Victim Count','Perpetrator Count']] = homicide_df[['Incident','Victim Count','Perpetrator Count']].astype(object)

homicide_df[['Perpetrator Age', 'Victim Count', 'Perpetrator Count']] = homicide_df[['Perpetrator Age', 'Victim Count', 'Perpetrator Count']].apply(pd.to_numeric, errors ='ignore')



#convert datatypes of columns to category

for col in ['City','State','Year','Month','Crime Type','Crime Solved','Victim Sex','Victim Race','Perpetrator Sex','Perpetrator Race','Relationship','Weapon']:

    homicide_df[col] = homicide_df[col].astype('category', ordered = True)

print(homicide_df.dtypes)

#prints categories

print(homicide_df['Victim Sex'].cat.categories)

#print numerical values

print(homicide_df['Victim Sex'].cat.codes.unique())
#concise summary of the dataframe

#we don't have missing data

print(homicide_df.info())
#check to see if any value is NaN in the dataframe

print('are there any null values?',homicide_df.isnull().values.any())

print('if there is string in numeric column',np.any([isinstance(val, str) for val in homicide_df['Perpetrator Age']]))
#print rows with str data type from Perpetrator Age which must have int type

def check_type(homicide_df,col):

    return homicide_df.loc[homicide_df[col].apply(type)==str,col]

print('Check Type for Perpetrator Age',check_type(homicide_df, 'Perpetrator Age'))
#convert data types of values other than int to NaN

homicide_df['Perpetrator Age']=pd.to_numeric(homicide_df['Perpetrator Age'], errors='coerce')

#confirm that they were converted to NaN

print('are there any null values?',homicide_df.isnull().values.any())

#convert data types of values from NaN to int

homicide_df['Perpetrator Age'] = homicide_df['Perpetrator Age'].fillna(0).astype(int)

#confirm that there are no longer NaNs

print('are there any null values?',homicide_df.isnull().values.any())

#confirm that str data types were converted to int

print('if there is string in numeric column',np.any([isinstance(val, str) for val in homicide_df['Perpetrator Age']]))
#generate descriptive statistics that summarize the central tendency, dispersion 

#and shape of the dataset’s distribution, excluding NaN values.

print(homicide_df.describe())
#show unique values used on the dataset

print('Crime Types = ', homicide_df['Crime Type'].unique())

print('Victime Races = ', homicide_df['Victim Race'].unique())

print('Perpetrator Races = ', homicide_df['Perpetrator Race'].unique())

print('Relationships = ', homicide_df['Relationship'].unique())

print('Weapons = ', homicide_df['Weapon'].unique())
#before checking and dropping rows, check whether all indices are of unique values

print('Are indices unique?',homicide_df.index.is_unique)
#check for outliers with the data



print('1980 < Year > 2014 - ',((homicide_df['Year'] > 2014) | (homicide_df['Year'] < 1980)).any())

clean_year = homicide_df[homicide_df['Year'] < 1980]

print(clean_year)



#oldest living person in USA recorded between 1980-2014 was 119 yrs old

print('Victim Age > 119 or Victim Age < 0 - ',((homicide_df['Victim Age'] > 119) | (homicide_df['Victim Age'] < 0)).any())

outlier_victim_age = homicide_df[homicide_df['Victim Age'] > 119]

print('Outliear data:\n', outlier_victim_age['Victim Age'].head())

homicide_df = homicide_df.drop(homicide_df[((homicide_df['Victim Age'] > 119) | (homicide_df['Victim Age'] < 0))].index)

print('Victim Age > 119 or Victim Age < 0 - ',((homicide_df['Victim Age'] > 119) | (homicide_df['Victim Age'] < 0)).any())

print('Perpetrator Age > 119  or Perpetrator Age < 8 - ',((homicide_df['Perpetrator Age'] > 119) | (homicide_df['Perpetrator Age'] < 8)).any())



#drop indices with Perpetrator Age < 8 since youngest recorded perpetrator for homicide between 1980-2014 was 8

homicide_df = homicide_df.drop(homicide_df[(homicide_df['Perpetrator Age'] < 8)].index)

print('Perpetrator Age < 8 - ',(homicide_df['Perpetrator Age'] < 8).any())
#generate descriptive statistics that summarize the central tendency, dispersion 

#and shape of the dataset’s distribution, excluding NaN values.

print(homicide_df.describe())
#count frequency of all data

cities = homicide_df['City'].value_counts().head(5)

states = homicide_df['State'].value_counts()

years = homicide_df['Year'].value_counts()

months = homicide_df['Month'].value_counts()

crime_types = homicide_df['Crime Type'].value_counts()

crime_solved = homicide_df['Crime Solved'].value_counts()

victim_sex = homicide_df['Victim Sex'].value_counts()

victim_ages = homicide_df['Victim Age'].value_counts()

victim_races = homicide_df['Victim Race'].value_counts()

perpetrator_sex = homicide_df['Perpetrator Sex'].value_counts()

perpetrator_ages = homicide_df['Perpetrator Age'].value_counts()

perpetrator_races = homicide_df['Perpetrator Race'].value_counts()

relationships = homicide_df['Relationship'].value_counts()

weapons = homicide_df['Weapon'].value_counts()
#show a graphical representation for the top and bottom 5 of data

def plot_frequency(x,data):

    plt.subplot(2, 1, 1)

    sns.countplot(x=x, data=data,

                              order=data[x].value_counts().iloc[:5].index)

    plt.subplot(2, 1, 2)

    sns.countplot(x=x, data=data,

                              order=data[x].value_counts().iloc[-5:].index)        

    plt.show()
plot_frequency('City',homicide_df)
plot_frequency('State',homicide_df)
plot_frequency('Year',homicide_df)
plot_frequency('Month',homicide_df)
plot_frequency('Crime Type',homicide_df)
plot_frequency('Crime Solved',homicide_df)
plot_frequency('Victim Sex',homicide_df)
plot_frequency('Victim Age',homicide_df)
plot_frequency('Victim Race',homicide_df)
plot_frequency('Perpetrator Sex',homicide_df)
plot_frequency('Perpetrator Age',homicide_df)
plot_frequency('Perpetrator Race',homicide_df)
plot_frequency('Relationship',homicide_df)
plot_frequency('Weapon',homicide_df)
#COMPARE VARIABLES

#show top 3 of x axis

year_crimesolved = sns.countplot(x='Year', data=homicide_df, hue='Crime Solved',

                                order=homicide_df.Year.value_counts().iloc[:3].index)
def crosstab(homicide_df,row, col):

    return(pd.crosstab(homicide_df[row], homicide_df[col]))

    

page_vage = crosstab(homicide_df,'Victim Age','Perpetrator Age')

month_solved = crosstab(homicide_df,'Month','Crime Solved')

year_solved = crosstab(homicide_df,'Year','Crime Solved')

year_month = crosstab(homicide_df,'Year','Month')

relationship_vsex = crosstab(homicide_df,'Relationship','Victim Sex')

state_solved = crosstab(homicide_df,'State','Crime Solved')

relationship_weapon = crosstab(homicide_df,'Relationship','Weapon')

weapon_psex = crosstab(homicide_df,'Weapon','Perpetrator Sex')

weapon_page = crosstab(homicide_df,'Perpetrator Age','Weapon')
def plot_heatmap(crosstab, annot,xtick, ytick, xrot, yrot):

    plt.figure(figsize=(8, 6))

    heatmap = sns.heatmap(crosstab,annot=annot,xticklabels=xtick, yticklabels=ytick)

    plt.xticks(rotation=xrot)

    plt.yticks(rotation=yrot)

    return heatmap
plot_heatmap(page_vage,False,10,10,0,0)
print(year_solved)

plot_heatmap(year_solved,True,True,True,0,0)
plot_heatmap(month_solved,True,True,True,0,0)
plot_heatmap(year_month,False,True,True,90,0)
plot_heatmap(relationship_vsex,True,True,True,0,0)
plot_heatmap(state_solved,False,True,True,0,0)
plot_heatmap(relationship_weapon,False,True,True,90,0)
plot_heatmap(weapon_psex,True,True,True,0,0)
plot_heatmap(weapon_page,False,True,10,90,0)