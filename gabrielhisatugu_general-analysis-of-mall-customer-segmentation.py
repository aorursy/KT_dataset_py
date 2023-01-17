import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [15, 15] # Defining the standard figure size

import seaborn as sns

import os
file = pd.read_csv('../input/Mall_Customers.csv')
file.head()
file.info()
file.isnull().any()
file['Age'].min()
file['Age'].max()
file.columns
file.rename(columns={'CustomerID':'ID','Annual Income (k$)':'Annual Income','Genre':'Gender'}, inplace=True)
file.describe()
sns.countplot(y='Age', data=file)

plt.figure(figsize=(20,20))

plt.show()
file_ages = file[['ID','Age']]

file_ages.head()
age = file_ages.groupby((file_ages.Age//10*10))['ID'].count()

plt.title('Customer density per age')

age.plot()

plt.show()

print(age)
sns.countplot(x='Gender', data=file)

plt.title('Customer gender density')

plt.show()

print(file.groupby(['Gender'])['ID'].count())
sns.lmplot(x='Spending Score (1-100)', 

           y='Age', data=file, 

           fit_reg=False, 

           hue='Gender')
sns.catplot(x='Spending Score (1-100)', 

            y='Age', 

            data=file, 

            col='Gender', 

            kind='swarm',

            hue='Gender')

x_axis = [0,50,100]

x_lab = ['0','50','100']

plt.xticks(x_axis,x_lab)

plt.xlim(-1, 100)
s = sns.FacetGrid(file, col='Gender', hue='Age')

s.map(plt.scatter, 'Spending Score (1-100)','Annual Income')

plt.xlim(0, 100)

plt.figure(figsize=(15, 15))
# Setting up colors to represent the decades in the plot

conditions = [

    (file['Age'] > 0) & (file['Age'] <= 20),

    (file['Age'] > 20) & (file['Age'] <= 30),

    (file['Age'] > 30) & (file['Age'] <= 40),

    (file['Age'] > 40) & (file['Age'] <= 50),

    (file['Age'] > 50) & (file['Age'] <= 60),

    (file['Age'] > 60)]

choices = ['20s', '30s', '40s', '50s', '60s', '70s']

file['Colors'] = np.select(conditions, choices, default='black')



sns.catplot(x='Spending Score (1-100)', 

            y='Annual Income', 

            data=file, 

            col='Gender', 

            kind='swarm',

            hue='Colors')

x_axis = [0,50,100]

x_lab = ['0','50','100']

plt.xticks(x_axis,x_lab)

plt.xlim(-1, 100)