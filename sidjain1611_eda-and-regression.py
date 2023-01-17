# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df= pd.read_csv('../input/world-happiness-report-2019.csv')
df.head()
df.describe(include='all')
df.isnull().sum()
plt.figure(figsize=(10,8))

# Correlation Map

corr = df.corr()

sns.heatmap(data=corr, square=True , annot=True)
df['Positive affect'].fillna(df['Positive affect'].mean(),inplace=True)

df['Negative affect'].fillna(df['Negative affect'].mean(),inplace=True)

df['Social support'].fillna(df['Social support'].mode(),inplace=True)

df['Freedom'].fillna(df['Freedom'].median(),inplace=True)

df['Corruption'].fillna(df['Corruption'].mean(),inplace=True)

df['Generosity'].fillna(df['Generosity'].median(),inplace=True)

df['Log of GDP\nper capita'].fillna(df['Log of GDP\nper capita'].mean(),inplace=True)

df['Healthy life\nexpectancy'].fillna(df['Healthy life\nexpectancy'].mean(),inplace=True)
asia = ["Israel", "United Arab Emirates", "Singapore", "Thailand", "Taiwan Province of China",

                 "Qatar", "Saudi Arabia", "Kuwait", "Bahrain", "Malaysia", "Uzbekistan", "Japan",

                 "South Korea", "Turkmenistan", "Kazakhstan", "Turkey", "Hong Kong S.A.R., China", "Philippines",

                 "Jordan", "China", "Pakistan", "Indonesia", "Azerbaijan", "Lebanon", "Vietnam",

                 "Tajikistan", "Bhutan", "Kyrgyzstan", "Nepal", "Mongolia", "Palestinian Territories",

                 "Iran", "Bangladesh", "Myanmar", "Iraq", "Sri Lanka", "Armenia", "India", "Georgia",

                 "Cambodia", "Afghanistan", "Yemen", "Syria"]

europe = ["Norway", "Denmark", "Iceland", "Switzerland", "Finland",

                 "Netherlands", "Sweden", "Austria", "Ireland", "Germany",

                 "Belgium", "Luxembourg", "United Kingdom", "Czech Republic",

                 "Malta", "France", "Spain", "Slovakia", "Poland", "Italy",

                 "Russia", "Lithuania", "Latvia", "Moldova", "Romania",

                 "Slovenia", "North Cyprus", "Cyprus", "Estonia", "Belarus",

                 "Serbia", "Hungary", "Croatia", "Kosovo", "Montenegro",

                 "Greece", "Portugal", "Bosnia and Herzegovina", "Macedonia",

                 "Bulgaria", "Albania", "Ukraine"]

north_america = ["Canada", "Costa Rica", "United States", "Mexico",  

                 "Panama","Trinidad and Tobago", "El Salvador", "Belize", "Guatemala",

                 "Jamaica", "Nicaragua", "Dominican Republic", "Honduras",

                 "Haiti"]

south_america = ["Chile", "Brazil", "Argentina", "Uruguay",

                 "Colombia", "Ecuador", "Bolivia", "Peru",

                 "Paraguay", "Venezuela"]

australia = ["New Zealand", "Australia"]

d_asia = dict.fromkeys(asia, 'Asia')

d_europe = dict.fromkeys(europe, 'Europe')

d_north_america = dict.fromkeys(north_america, 'North America')

d_south_america = dict.fromkeys(south_america, 'South America')

d_australia = dict.fromkeys(australia, 'Australia')

continent_dict = {**d_asia, **d_europe, **d_north_america, **d_south_america, **d_australia}

df["continent"] = df["Country (region)"].map(continent_dict)

df.continent.fillna("Africa", inplace=True)
sns.barplot(x="Healthy life\nexpectancy", y="continent", data=df, palette='Accent')
sns.barplot(x="Ladder", y="continent", data=df, palette='Accent')
print(df[['Country (region)', 'Healthy life\nexpectancy']].groupby('Country (region)').mean().sort_values('Healthy life\nexpectancy', ascending=False).head(10))
most_happy = df.sort_values('Healthy life\nexpectancy', ascending = False).head(10)

plt.figure(figsize=(20,8))

sns.barplot(most_happy['Country (region)'], most_happy['Healthy life\nexpectancy'], palette='Accent')
print(df[['Country (region)', 'Healthy life\nexpectancy']].groupby('Country (region)').mean().sort_values('Healthy life\nexpectancy', ascending=False).tail(10))
most_happy = df.sort_values('Healthy life\nexpectancy', ascending = False).tail(10)

plt.figure(figsize=(20,8))

sns.barplot(most_happy['Country (region)'], most_happy['Healthy life\nexpectancy'], palette='Accent')
country_wise = df[['Country (region)', 'Healthy life\nexpectancy']]

country_wise.plot(kind = 'line',figsize=(20,8),color='g')

plt.show()
sns.scatterplot(x='Ladder',y='Healthy life\nexpectancy',data=df)
sns.scatterplot(x='Log of GDP\nper capita',y='Healthy life\nexpectancy',data=df)
most_gdp = df.sort_values('Log of GDP\nper capita', ascending = False).head(10)

plt.figure(figsize=(20,8))

sns.barplot(most_gdp['Country (region)'], most_gdp['Log of GDP\nper capita'], palette='Accent')
sns.scatterplot(x='Social support',y='Healthy life\nexpectancy',data=df)
df=df.drop('Country (region)',axis=1)

df=df.drop('continent',axis=1)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X=df.iloc[:,:-1].values

y=df.iloc[:,:11].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)
plt.scatter(y_test, y_pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted y')
# Evaluate the data

from sklearn import metrics

print("MAE", metrics.mean_absolute_error(y_test, y_pred))

print("MSE", metrics.mean_squared_error(y_test, y_pred))

print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn import metrics

metrics.r2_score(y_test,y_pred)