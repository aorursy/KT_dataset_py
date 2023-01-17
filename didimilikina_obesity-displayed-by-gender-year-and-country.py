# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = '/kaggle/input/obesity-among-adults-by-country-19752016/obesity-cleaned.csv'



df = pd.read_csv(data)



df.isnull().sum()
df.head(10)
df.tail(10)
df = df.drop(columns=['Unnamed: 0'])
df.rename(columns={'Obesity (%)' : 'Obesity_percentage'}, inplace=True)

df['Obesity_percentage'] = df['Obesity_percentage'].apply(lambda x: (x.split()[0]))



df_clean = df[df["Obesity_percentage"] != "No"]
df_clean.info()
df_clean['Obesity_percentage'] = df_clean['Obesity_percentage'].apply(lambda x: float(x))
df_clean.head(3)
both_genders = df_clean[df_clean["Sex"] == "Both sexes"].groupby("Year").Obesity_percentage.mean()

male = df_clean[df_clean["Sex"] == "Male"].groupby("Year").Obesity_percentage.mean()

female = df_clean[df_clean["Sex"] == "Female"].groupby("Year").Obesity_percentage.mean()
plt.figure(figsize=(20,7))



plt.plot(both_genders, linestyle='dashed', label="Obesity percentage - Both Genders", color='green')

plt.plot(male, linestyle='dotted', label="Obesity percentage - Males", color='blue')

plt.plot(female, linestyle='solid', label="Obesity percentage - Females",color='red')



plt.xlabel('Year', fontsize=17)

plt.ylabel('Obesity percentage', fontsize=17)

plt.title('Mean Obesity by Year', fontsize=20)



plt.grid(True)

plt.legend()

plt.tight_layout()
df_clean.Country.unique()
fig = plt.figure(figsize=(10,10))

 

top_obese_countries = df_clean[(df_clean["Year"]==2016) & (df_clean["Sex"]=="Male") & (df_clean["Obesity_percentage"] > 33)].groupby("Country").Obesity_percentage.sum().sort_values(ascending=False)



top_obese_countries.plot(kind="bar",title='Countries with 1/4 percent male obesity in 2016', fontsize=20)
fig = plt.figure(figsize=(10,10))

 

top_obese_countries = df_clean[(df_clean["Year"]==1975) & (df_clean["Sex"]=="Male") & (df_clean["Obesity_percentage"] > 33)].groupby("Country").Obesity_percentage.sum().sort_values(ascending=False)



top_obese_countries.plot(kind="bar",title='Countries with 1/4 percent male obesity in 1975', fontsize=20)
fig = plt.figure(figsize=(15,10))

 

top_obese_countries = df_clean[(df_clean["Year"]==2016) & (df_clean["Sex"]=="Female") & (df_clean["Obesity_percentage"] > 33)].groupby("Country").Obesity_percentage.sum().sort_values(ascending=False)



top_obese_countries.plot(kind="bar",title='Countries with 1/4 percent female obesity in 2016', fontsize=20)
fig = plt.figure(figsize=(10,10))

 

top_obese_countries = df_clean[(df_clean["Year"]==1975) & (df_clean["Sex"]=="Female") & (df_clean["Obesity_percentage"] > 33)].groupby("Country").Obesity_percentage.sum().sort_values(ascending=False)



top_obese_countries.plot(kind="bar",title='Countries with 1/4 percent female obesity in 1975', fontsize=20)
fig = plt.figure(figsize=(10,10))

 

top_obese_countries = df_clean[(df_clean["Year"]==2016) & (df_clean["Sex"]=="Both sexes") & (df_clean["Obesity_percentage"] > 33)].groupby("Country").Obesity_percentage.sum().sort_values(ascending=False)



top_obese_countries.plot(kind="bar",title='Countries with 1/4 percent both genders obesity in 2016', fontsize=20)