# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from scipy import stats

from scipy.stats import norm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



# display settings

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)



# filter warnings

import warnings

warnings.filterwarnings("ignore")
# reading from csv and creating dataframe

df = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
# dimensions of the dataframe

print("No. of rows: {}\t\tNo. of columns: {}".format(*df.shape))
# columns info

df.info()
# printing first 5 rows

df.head()
# printing last 5 rows

df.tail()
# dropping the Serial No. column

df.drop(columns=['Serial No.'], inplace=True)
# descriptive statistics

df.describe().T
# column names

list(df.columns)
# removing the trailing spaces in column names

df.rename(columns={'LOR ':'LOR',

                  'Chance of Admit ':'Chance of Admit'}, inplace=True)
# % of missing values in each column

(df.isna().sum() / df.shape[0]) * 100
# observing for skew or non-linearness

print("Acceptable skew range -1 to 1:", df['Chance of Admit'].skew())



# deviation of data from normal distribution using QQ plot

fig, ax = plt.subplots(1,2, figsize=(15,6))



mu, sigma = norm.fit(df['Chance of Admit'])

sns.distplot(df['Chance of Admit'], fit=norm, ax=ax[0])

res = stats.probplot(df['Chance of Admit'], plot=ax[1])

plt.show()
print(df['GRE Score'].describe())



plt.subplots(1,2, figsize=(15,5))



# checking for outliers

plt.subplot(1,2,1)

sns.boxplot(df['GRE Score'])

plt.title("Boxplot to identify the outliers")



# checking the distribution

plt.subplot(1,2,2)

sns.distplot(df['GRE Score'])

plt.title("Distplot to check the skewness")



plt.show()
print(df['TOEFL Score'].describe())



plt.subplots(1,2, figsize=(15,5))



# checking for outliers

plt.subplot(1,2,1)

sns.boxplot(df['TOEFL Score'])

plt.title("Boxplot to identify the outliers")



# checking the distribution

plt.subplot(1,2,2)

sns.distplot(df['TOEFL Score'])

plt.title("Distplot to check the skewness")



plt.show()
print(df['University Rating'].describe())



plt.subplots(1,2, figsize=(15,5))



# checking for outliers

plt.subplot(1,2,1)

sns.boxplot(df['University Rating'])

plt.title("Boxplot to identify the outliers")



# checking the distribution

plt.subplot(1,2,2)

sns.distplot(df['University Rating'])

plt.title("Distplot to check the skewness")



plt.show()
print(df['SOP'].describe())



plt.subplots(1,2, figsize=(15,5))



# checking for outliers

plt.subplot(1,2,1)

sns.boxplot(df['SOP'])

plt.title("Boxplot to identify the outliers")



# checking the distribution

plt.subplot(1,2,2)

sns.distplot(df['SOP'])

plt.title("Distplot to check the skewness")



plt.show()
print(df['LOR'].describe())



plt.subplots(1,2, figsize=(15,5))



# checking for outliers

plt.subplot(1,2,1)

sns.boxplot(df['LOR'])

plt.title("Boxplot to identify the outliers")



# checking the distribution

plt.subplot(1,2,2)

sns.distplot(df['LOR'])

plt.title("Distplot to check the skewness")



plt.show()
# removing outliers

Q1 = df['LOR'].quantile(0.25); Q3 = df['LOR'].quantile(0.75); IQR = Q3 - Q1

df['LOR'].loc[((df['LOR'] < Q1-1.5*IQR) | (df['LOR'] > Q3+1.5*IQR))] = np.NaN



# filling missing values

df['LOR'] = df['LOR'].fillna(df['LOR'].median())
print(df['CGPA'].describe())



plt.subplots(1,2, figsize=(15,5))



# checking for outliers

plt.subplot(1,2,1)

sns.boxplot(df['CGPA'])

plt.title("Boxplot to identify the outliers")



# checking the distribution

plt.subplot(1,2,2)

sns.distplot(df['CGPA'])

plt.title("Distplot to check the skewness")



plt.show()
print(df['Research'].describe())



plt.subplots(1,2, figsize=(15,5))



# checking for outliers

plt.subplot(1,2,1)

sns.boxplot(df['Research'])

plt.title("Boxplot to identify the outliers")



# checking the distribution

plt.subplot(1,2,2)

sns.distplot(df['Research'])

plt.title("Distplot to check the skewness")



plt.show()
# checking the variables info at the end of data pre-processing and cleaning

df.info()
# creating a class for top 75% of students

Q3 = df['Chance of Admit'].quantile(0.75)

df['High Chance'] = df['Chance of Admit'].apply(lambda x: 1 if x > Q3 else 0)



df['High Chance'].value_counts()
# checking variables info at the end of feature engineering

df.info()
# statistical summary

print("Statistical summary:\n", df['GRE Score'].describe())



# distribution of data

fig, ax = plt.subplots(1,2, figsize=(15,6))

sns.distplot(df['GRE Score'], bins=150, ax=ax[0])

ax[0].set_title("GRE Score")



# distribution of data of top 25% of students w.r.t. chance of admit

sns.distplot(df[df["High Chance"] == 1]['GRE Score'], bins=100, ax=ax[1])

ax[1].set_title("GRE Score of top 25%")

plt.show()
# statistical summary

print("Statistical summary:\n", df["TOEFL Score"].describe())



# distribution of data

fig, ax = plt.subplots(1,2, figsize=(15,6))

sns.distplot(df["TOEFL Score"], bins=150, ax=ax[0])

ax[0].set_title("TOEFL Score")



# distribution of data of top 25% of students w.r.t. chance of admit

sns.distplot(df[df["High Chance"] == 1]["TOEFL Score"], bins=100, ax=ax[1])

ax[1].set_title("TOEFL Score of top 25% of students")

plt.show()
# count plot of each rating

fig, ax = plt.subplots(1,2, figsize=(15,6))

sns.countplot(df['University Rating'], ax=ax[0])

ax[0].set_title("Count plot per Rating")



# percentage proportions

ax[1].pie(df['University Rating'].value_counts(), labels=df['University Rating'].value_counts().index, autopct='%1.1f',

         explode=[0,0,0,0.1,0])

ax[1].set_title("Percentage proportion of each Rating")

plt.show()
# count plot of each SOP rating

fig, ax = plt.subplots(1,2, figsize=(15,6))

sns.countplot(df["SOP"], ax=ax[0])

ax[0].set_title("Count plot of SOP ratings")



# count of SOP rating for top 25% of students w.r.t. chance of admit

sns.countplot(df[df["High Chance"] == 1]["SOP"], ax=ax[1])

ax[1].set_title("Count plot of SOP ratings of top 25% students")

plt.show()
# count plot of each LOR rating

fig, ax = plt.subplots(1,2, figsize=(15,6))

sns.countplot(df["LOR"], ax=ax[0])

ax[0].set_title("Count plot of LOR ratings")



# count of LOR rating for top 25% of students w.r.t. chance of admit

sns.countplot(df[df["High Chance"] == 1]["LOR"], ax=ax[1])

ax[1].set_title("Count plot of LOR ratings of top 25% students")

plt.show()
# statistical summary

print("Statistical summary:\n", df["CGPA"].describe())



# distribution of data

fig, ax = plt.subplots(1,2, figsize=(15,6))

sns.distplot(df["CGPA"], bins=75, ax=ax[0])

ax[0].set_title("CGPA")



# distribution of data of top 25% of students w.r.t. chance of admit

sns.distplot(df[df["High Chance"] == 1]["CGPA"], bins=75, ax=ax[1])

ax[1].set_title("CGPA of top 25% of students")

plt.show()
# no. of observations

print(df["Research"].value_counts())



# count plot of each obersations

fig, ax = plt.subplots(1,3, figsize=(18,6))

sns.countplot(df["Research"], ax=ax[0])

ax[0].set_title("No. of students with and without research")



# pie chart of percentage proportion

ax[1].pie(df["Research"].value_counts(), labels=df["Research"].value_counts().index, autopct="%1.1f")

ax[1].set_title("Percentage proportion")



# pie chart of percentage proportion for top 25% students w.r.t. chance of admit

ax[2].pie(df[df["High Chance"] == 1]["Research"].value_counts(), labels=df[df["High Chance"] == 1]["Research"].value_counts().index,

         autopct="%1.1f", explode=[0,0.1])

ax[2].set_title("Percentage proportion of top 25% students")

plt.show()
# heat map

plt.figure(figsize=(10,8))

sns.heatmap(df.corr(), annot=True, cmap="coolwarm", mask=np.triu(df.corr(), 1))

plt.show()
# pair plot

sns.pairplot(df, hue="High Chance")

plt.show()
# scatter plots

fig, ax = plt.subplots(3,1, figsize=(8,20))

sns.scatterplot(data=df, x="CGPA", y="Chance of Admit", hue="High Chance", ax=ax[0])

ax[0].set_title("GRE Score vs Chance of Admit")



sns.scatterplot(data=df, x="GRE Score", y="Chance of Admit", hue="High Chance", ax=ax[1])

ax[0].set_title("GRE Score vs Chance of Admit")



sns.scatterplot(data=df, x="TOEFL Score", y="Chance of Admit", hue="High Chance", ax=ax[2])

ax[0].set_title("GRE Score vs Chance of Admit")



plt.show()
# violin plots

fig, ax = plt.subplots(1,2, figsize=(15,6))

sns.violinplot(data=df, x="SOP", y="Chance of Admit", hue="High Chance", split=True, ax=ax[0])

ax[0].set_title("SOP vs Chance of Admit")



sns.violinplot(data=df, x="LOR", y="Chance of Admit", hue="High Chance", split=True, ax=ax[1])

ax[1].set_title("LOR vs Chance of Admit")



plt.show()
# proportional plot

ct_highchance_research = pd.crosstab(df["High Chance"], df["Research"], normalize=0)

print(ct_highchance_research)



# stacked bar chart to show the proportion of students with and without reasearch experience  

fig, ax = plt.subplots(1,2, figsize=(12,6))

ct_highchance_research.plot.bar(stacked=True, ax=ax[0])

ax[0].set_title("Stacked bar chart")



# bar plot with mean value of Chance of Admit for each research category

sns.barplot(data=df, x="Research", y="Chance of Admit", ax=ax[1])

ax[1].set_title("Bar plot with mean value of Chance of Admit")



plt.show()
# columns info

df.info()
# creating dependent variable

y = df["Chance of Admit"]



# creating independent variables

X = df.drop(columns=["Chance of Admit","High Chance"])



X.head()
# creating list of categorical 

list_categorical_columns = ["Research"]

print("Categorical variables:\t", list_categorical_columns)



# creating list of numeric variable

list_numeric_columns = list(X.drop(columns="Research").columns)

print("Numeric variables:\t", list_numeric_columns)
# getting the unique values for categorical variables

for col in list_categorical_columns:

    print(col)

    print(df[col].value_counts())

    print()
# changing the datatype to uint8

X["Research"] = X["Research"].astype("uint8")
# splititng my data into 80:20 split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# initializing scaler and fitting with train data

ss = StandardScaler().fit(X_train[list_numeric_columns])



# transforming train data

X_train[list_numeric_columns] = ss.transform(X_train[list_numeric_columns])



# transforming test data

X_test[list_numeric_columns] = ss.transform(X_test[list_numeric_columns])
# initializing and fitting the model

model_lr = LinearRegression().fit(X_train,y_train)



# predicting using training data

y_pred_train = model_lr.predict(X_train)



# predicting using test data

y_pred_test = model_lr.predict(X_test)
# lets compare model performance on trainig and test data to check for overfitting

print("Train vs Test model performance results:")

pd.DataFrame(zip([np.sqrt(mean_squared_error(y_pred_train,y_train)), np.sqrt(mean_squared_error(y_pred_test,y_test))],

                 [r2_score(y_pred_train,y_train), r2_score(y_pred_test,y_test)]), columns=["RMSE", "R-squared"], index=["Train","Test"])