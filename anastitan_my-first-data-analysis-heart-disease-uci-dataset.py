# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Reading the original csv file

df = pd.read_csv("../input/heart.csv")
df.columns #View the column headers
df.head()  #Peek into the first 5 rows
df.tail()  #Peek into the last 5 rows
df.shape   #Look at the Shape of the DataFrame
df.info()  #Look at the data types and look for null values using info()
df.describe() # Descriptive statistics (or) Summary Statistics 
# Sorting the DataFrame by Age

print(df.sort_values(by=['age']))
df['sex'] = df['sex'].astype('category') # convert the sex column as categorical using astype() method
# Distribution of Age using Histogram

plt.hist(df.age, bins=10, color='green')

plt.xlabel('AGE')

plt.ylabel('Frequency')

plt.title('Histogram of Age')

plt.show()
# Plotting the Rest Blood Pressure against age and comparison between sex

sns.lmplot(x='age', y='trestbps', data=df, hue='sex', palette='muted')

plt.show()

print('Correlation Coefficient:{}'.format(np.corrcoef(df.age, df.trestbps)[0,1]))
# BeeSwarm plot for classifying common kind of heart defects in males and females

sns.swarmplot(x='sex', y='thal', data=df, size=7)

plt.show()
# Heatmap for itendifying the correlation between different variables

corrmat = df.corr()

f, ax = plt.subplots(figsize=(10,9))

sns.heatmap(corrmat, square=True, annot = True, annot_kws={'size':10})

plt.show()
# Regression model between Resting Blood Pressure and Age and effect of blood sugar level of them

sns.lmplot(x='age', y='trestbps', data=df, hue='fbs')

plt.show()
# Pair plot between ordinal variables

sns.set()

cols1= ['slope','cp', 'thalach', 'target']

sns.pairplot(df[cols1], size=2)

plt.show()
# filtering of DataFrame using Boolean functions

df1 = df[(df['sex']==1) & (df['fbs']==1)]

print(df1)
# Usage of groupby function # Multi-leve grouping

print("Average Cholestrol based on Gender")

print(df.groupby('sex')['chol'].mean())

print("Average Resting Blood Pressure based on Gender")

print(df.groupby('sex')['trestbps'].mean())
# Grouping using aggregate method

df.groupby(['thal','cp']).mean()
# Violin Plots to demonstrate the highest heart rate achieved and cholestrol for different chest pain types

plt.figure(figsize=(10,7))

plt.subplot(3,1,1)

sns.violinplot(x='cp', y='thalach', inner='points', data=df)

plt.xticks

plt.subplot(3,1,2)

sns.violinplot(x='cp', y='chol', inner='points', data=df)

plt.subplot(3,1,3)

sns.violinplot(x='cp', y='trestbps', inner='points', data=df)

plt.tight_layout()

plt.show()
# Beeswarm plot for understanding the defect types for different ages in males and females

sns.swarmplot(x='sex', y='age', hue='thal', data=df, size=7, palette='deep')

plt.legend(title='thal', loc='lower center')

plt.show()
# Beeswarm Plot to understand the heart failures in males and females with chest pain types

sns.swarmplot(x='cp', y='trestbps', hue='target', data=df)

plt.title("0: Female , 1: Male")

plt.show()
# Lambda Function to normalize the cholestrol and resting BP between 0 and 1

normalize = lambda col_name: df[col_name] /df[col_name].max()

df['trestbps_norm'] = normalize('trestbps')

df['chol_norm'] = normalize('chol')
# Joint Plot with Contours for cholestrol levels for age groups

sns.jointplot(x='age', y='chol_norm', data=df, kind='kde')

plt.show()
# Multiple plots using subplot

plt.figure(figsize=(10,8))

plt.subplot(3,1,1)

plt.plot(df['trestbps'])

plt.subplot(3,1,2)

plt.plot(df['chol'])

plt.subplot(3,1,3)

plt.plot(df['thalach'])

plt.tight_layout()

plt.show()
# Empirical cumulative distribution function to understand the distribution of Resting BP, Cholestrol and Max. HR Ach.

def ecdf(data):

    """

    Function Definition: Empirical cumulative distribution function 

    to understand the distribution of Resting BP, Cholestrol and 

    Max. HR Ach.

    

    """

    n=len(data)

    x=np.sort(data)

    y=np.arange(1, n+1)/n

    return x, y

x1, y1 = ecdf(df['trestbps'])

plt.plot(x1, y1, marker='.', linestyle='none')

plt.xlabel('Resting Blood Pressure')

plt.ylabel('ECDF')

plt.title('Empirical Cumulative Distribution Functions')

plt.margins(0.02)

plt.show()

x2, y2 = ecdf(df['chol'])

plt.plot(x2, y2, marker='.', linestyle='none')

plt.xlabel('Cholestrol')

plt.ylabel('ECDF')

plt.margins(0.02)

plt.show()

x3, y3 = ecdf(df['thalach'])

plt.plot(x3, y3, marker='.', linestyle='none')

plt.xlabel('Max. HR Achieved')

plt.ylabel('ECDF')

plt.margins(0.02)

plt.show()

x4, y4 = ecdf(df['oldpeak'])

plt.plot(x4, y4, marker='.', linestyle='none')

plt.xlabel('ST depression induced by exercise relative to rest')

plt.ylabel('ECDF')

plt.margins(0.02)

plt.show()
# Supervised learning ML model using scikit learning module

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

X = df.drop('target', axis=1).values

y = df['target'].values

knn = KNeighborsClassifier(n_neighbors=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Test Set Predictions:\n{}".format(y_pred))

knn.score(X_test, y_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
