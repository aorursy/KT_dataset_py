# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
df.head()
df.tail()
df.info()
df.describe()
df.dtypes
check_missing_data = df.isnull()

check_missing_data
for column in check_missing_data.columns.values.tolist():

    print(column)

    print(check_missing_data[column].value_counts())

    print("")
df['TotalScore'] = (df['math score']+ df['reading score'] + df['writing score'])/3

df.head()
import seaborn as sns

import matplotlib.pyplot as plt
sns.boxplot(x="gender", y="math score", data=df)
sns.boxplot(x="gender", y="reading score", data=df)
sns.boxplot(x="gender", y="writing score", data=df)
sns.boxplot(x="gender", y="TotalScore", data=df)
df_gpby = df[['parental level of education','race/ethnicity','TotalScore']]

grouped_test1 = df_gpby.groupby(['parental level of education','race/ethnicity'],as_index=False).mean()

grouped_test1
grouped_pivot = grouped_test1.pivot(index='race/ethnicity', columns='parental level of education')

grouped_pivot
plt.pcolor(grouped_pivot, cmap='RdBu')

plt.colorbar()

plt.show()
fig, ax = plt.subplots()

im = ax.pcolor(grouped_pivot, cmap='RdBu')



#label names

row_labels = grouped_pivot.columns.levels[1]

col_labels = grouped_pivot.index



#move ticks and labels to the center

ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)

ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)



#insert labels

ax.set_xticklabels(row_labels, minor=False)

ax.set_yticklabels(col_labels, minor=False)



#rotate label if too long

plt.xticks(rotation=90)



fig.colorbar(im)

plt.show()
sns.boxplot(x="race/ethnicity", y="TotalScore", data=df)
sns.boxplot(x="parental level of education", y="TotalScore", data=df)

plt.xticks(rotation=90)
from scipy import stats
df_correlation = df[['math score','reading score' ]]

df_correlation



sns.regplot(x="reading score", y="math score", data=df_correlation)
pearson_coef, p_value = stats.pearsonr(df['math score'], df['reading score'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
df_correlation = df[['math score','writing score' ]]

df_correlation



sns.regplot(x="writing score", y="math score", data=df_correlation)
pearson_coef, p_value = stats.pearsonr(df['math score'], df['writing score'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
df_correlation = df[['reading score','writing score' ]]

df_correlation



sns.regplot(x="reading score", y="writing score", data=df_correlation)
pearson_coef, p_value = stats.pearsonr(df['writing score'], df['reading score'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
plt.figure(figsize=(14,4))



plt.subplot(1,4,1)

sns.barplot(x = 'lunch', y = 'reading score', data = df)



plt.subplot(1,4,2)

sns.barplot(x = 'lunch', y = 'writing score', data = df)



plt.subplot(1,4,3)

sns.barplot(x = 'lunch', y = 'math score', data = df)



plt.subplot(1,4,4)

sns.barplot(x = 'lunch', y = 'TotalScore', data = df)





plt.tight_layout()
plt.figure(figsize=(14,4))



plt.subplot(1,4,1)

sns.barplot(x = 'test preparation course', y = 'reading score', data = df)



plt.subplot(1,4,2)

sns.barplot(x = 'test preparation course', y = 'writing score', data = df)



plt.subplot(1,4,3)

sns.barplot(x = 'test preparation course', y = 'math score', data = df)



plt.subplot(1,4,4)

sns.barplot(x = 'test preparation course', y = 'TotalScore', data = df)





plt.tight_layout()
sns.boxplot(x="lunch", y="TotalScore", data=df)
sns.boxplot(x="test preparation course", y="TotalScore", data=df)
df['genderdummy'] = np.where(df.gender=='female',0,1)

df['lunchdummy'] = np.where(df.lunch == 'standard',1,0)

df['coursedummy'] = np.where(df['test preparation course']=='none',0,1)
df.head()
assigndegree = {"bachelor's degree":3, 'some college':4, "master's degree":5,

       "associate's degree":6, 'high school':1, 'some high school':2}

df['degreedummy'] = df['parental level of education'].map(assigndegree)

assignrace = {'group B':2, 'group C':3, 'group A':1, 'group D':4, 'group E':5}

df['racedummy'] = df['race/ethnicity'].map(assignrace)
df.head()
from sklearn.linear_model import LinearRegression
z = df[['racedummy','degreedummy', 'genderdummy', 'lunchdummy' , 'coursedummy']]

lm = LinearRegression()

lm.fit(z, df['TotalScore'])

lm.intercept_

lm.coef_
from sklearn.model_selection import train_test_split

X = df.drop(['TotalScore', 'parental level of education', 'race/ethnicity','test preparation course',

                   'lunch', 'gender', 'math score', 'reading score', 'writing score'], axis = 1)

Y = df['TotalScore']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
print("number of test samples :", x_test.shape[0])

print("number of training samples:",x_train.shape[0])
x_train.head()
from sklearn.linear_model import LinearRegression

lm1= LinearRegression()

lm1.fit(x_train,y_train)

lm1.score(x_test,y_test)