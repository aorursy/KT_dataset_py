import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn')

import os

#missingno is a python library enables a quick visual summary of the completeness of your dataset.
import missingno as msno
df = pd.read_csv('../input/titanic/train.csv')
#Check if any of the columns which are supposed to be numerical types are now Object type. The existence of missing values makes numerical columns into Object dtype.
df.info()
#Find the columns of Object dtype
object_columns = [c for c in df.columns if df[c].dtype == 'object']
object_columns
#Get the number of uniuqe values of each Object columns
df[object_columns].describe()
#Look for common symbols for missing values, like '?', 'NA', '-', '.', 'unknown'
i=1
for c in object_columns:
    print(str(i)+'.', "Top 10 unique values for column '{}':".format(c))
    unique_values = df[c].dropna().unique()
    print(np.sort(unique_values)[:10], '\n')
    print("Is'NA' one of its unique values?", 'NA' in unique_values, '\n')
    unique_values_lowercase = [v.lower() for v in unique_values]
    print("Is 'unknown' (case insensitive) one of its unique values?", 'unknown' in unique_values_lowercase, '\n')
    i+=1
#Look for values out of expected data range, like 0 for age, or extremely large number like 9999 for age
df.describe()
df[df.Fare==0]
# Amount of missingness
df.isnull().sum()
#Percentage of missingness
df.isnull().mean()*100
#Check nullity matrix for the columns with missingness and the column used to sort the data
df_sorted = df.sort_values(by='Survived')
cols_missing = ['Cabin', 'Age', 'Embarked']
cols_matrix = cols_missing.copy()
cols_matrix.append('Survived')
msno.matrix(df_sorted[cols_matrix], figsize=(5, 5), sparkline=False);
def missingness_by_value(col1, col2, df=df):
    """
    check the missingness (%) of a colomn varied by the values of another column
    
    Args:
    col1 (str): the name of a column with missing values
    col2 (str): the name of a column of which the values may affect the missingness of col1
    df (pd.DataFrame) : the dataframe contains both col1 and col2
    """
    for v in np.sort(df[col2].dropna().unique()):
        missingness = df.loc[df[col2]==v, col1].isnull().mean()*100
        print("Missingness percentage of {} when the value of {} equals {}: {}%".format(col1, col2, v, round(missingness)))
#check the missingness of 'Cabin' column varied by the values of 'Survived' column
missingness_by_value('Cabin', 'Survived')
#check if the variables' values affect the missingness of 'Cabin' or 'Age'
for c in df.columns:
    if c=='Survived':
        continue
    df_sorted = df.sort_values(by=c)
    cols_matrix = ['Cabin', 'Age']
    cols_matrix.append(c)
    print("Nullity matrix for data sorted by column '{}'".format(c))
    msno.matrix(df_sorted[cols_matrix], figsize=(5, 5), sparkline=False)
    plt.show()
missingness_by_value('Cabin', 'Pclass')
np.sort(df['Ticket'].unique())
df['Pure_Digit_Ticket'] = df['Ticket'].apply(str.isdigit)
missingness_by_value('Cabin', 'Pure_Digit_Ticket', df)
np.sort(df['Name'].unique())
missingness_by_value('Age', 'Embarked')
msno.heatmap(df[['Age', 'Cabin']], figsize=(5,5));
