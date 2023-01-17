# importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime # datetime processing



# Data visualisation

import matplotlib.pyplot as plt

import seaborn as sns



# setting path of the dataset

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
os.chdir('../input')

df_nation=pd.read_csv("nation_level_daily.csv")

df_nation
df_state=pd.read_csv("state_level_latest1.csv")

df_state

df_age=pd.read_csv("AgeRangeCount.csv")

df = pd.read_csv("patients_data1.csv")

# checking the columns

df.head()
df
df.shape
df.info()
unique_val = []

for i in df.columns:

    u = df[i].nunique()

    unique_val.append(u)

    

pd.DataFrame({"No. of unique values": unique_val}, index=df.columns)
# Working dataset

dataset = df.copy()

# plot of missing value

plt.figure(figsize=(9,5))

sns.heatmap(dataset.isnull(),yticklabels=False, cbar=False, cmap="Paired");

plt.title("Heatmap of Missing Values");
# Features with missing values

miss = dataset.isnull().sum().sort_values(ascending = False).head(20)

miss_per = (miss/len(dataset))*100



# Percentage of missing values

pd.DataFrame({'No. missing values': miss, '% of missind data': miss_per.values})
dataset['Gender'].value_counts()
# Filling null value

dataset['Gender'].fillna('Unknown', inplace = True)



# confimation after filling the null values

print("Null values before replacement :", df['Gender'].isnull().sum())

print("Null values after replacement :", dataset['Gender'].isnull().sum())
# Filling the null value 

dataset_age=dataset['AgeBracket'].fillna('Unknown', inplace=True)



# confimation after filling the null values

print("Null values before replacement :", df['AgeBracket'].isnull().sum())

print("Null values after replacement :", dataset['AgeBracket'].isnull().sum())
# Filling the null value 

dataset['Detected District'].fillna('unknown', inplace = True)



# confimation after filling the null values

print("Null values before replacement :", df['Detected District'].isnull().sum())

print("Null values after replacement :", dataset['Detected District'].isnull().sum())
state_new=df_state.loc[1:,['State','Confirmed','Recovered','Deaths','Active','Recovery_rate','Mortality_rate']]

state_new
def plot_pie_charts(x, y, title):

    c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_states))

    plt.figure(figsize=(20,15))

    plt.title(title, size=20)

    plt.pie(y, colors=c)

    plt.legend(x, loc='best', fontsize=15)

    plt.show()
plt.figure(figsize=(50, 50))

sns.barplot(x = 'State', y = "Confirmed", data=state_new);

plt.ylabel("Confirmed Covid cases");
# checking for top 10 affected areas/ Red zone

dataset['Current Status'].value_counts().head(15).plot(kind='bar', color='green');



# plot for top seller

plt.title("Current patient status", fontsize=14);

# checking for top 10 affected areas/ Red zone

dataset['Detected City'].value_counts().head(15).plot(kind='bar', color='brown');



# plot for top seller

plt.title("The most affected cities", fontsize=14);

# checking for top 10 affected areas/ Red zone

dataset['Date Announced'].value_counts().head(25).plot(kind='bar', color='blue');



# plot for top seller

plt.title("The most affected dates", fontsize=14);

plt.figure(figsize=(35, 15))

sns.barplot(x = 'State', y = "Recovery_rate", data=df_state);

plt.ylabel("Recovery Rate");
plt.figure(figsize=(45, 35))

sns.barplot(x = 'State', y = "Mortality_rate", data=df_state, palette='Set3');

plt.ylabel("Mortality Rate");



plt.figure(figsize=(10, 10))

sns.barplot(x = 'Age_Range', y = "Count", data=df_age);

plt.ylabel("Patient Number");
plt.figure(figsize=(15, 12))

sns.lineplot(x = 'Gender', y = "Patient Number", data=df);

plt.ylabel("Number of Patients");
corr=df.corr()

corr = (corr)

plt.figure(figsize=(10,5))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws=



{'size': 10},

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.scatterplot(x='Daily Recovered', y='Daily Confirmed', data=df_nation);

plt.ylabel("confirmed cases");
sns.regplot(x='Daily Recovered',y='Daily Confirmed',data=df_nation);
state_new['Recovery_rate']. plot(kind='hist');

state_new['Mortality_rate']. plot(kind='hist');


df_state['Confirmed']. plot(kind='hist');

df_state['Recovered']. plot(kind='hist');