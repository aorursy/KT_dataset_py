import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns



from  sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

#from sklearn.feature_selection import SelectKBest, chi2

from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")

df.head()
df.describe()
df.isnull().sum()
# Gre Score

plt.figure(figsize=(8,5))

sns.distplot(df['GRE Score'])

plt.title('GRE Score distribution')

plt.show()



plt.figure(figsize=(8,5))

sns.boxplot(df['GRE Score'])

plt.title('GRE Score distribution')

plt.show()
q = df['GRE Score'].quantile(0.01)

df = df[df['GRE Score']>q]

df.describe()
# Gre Score

plt.figure(figsize=(8,5))

sns.distplot(df['GRE Score'])

plt.title('GRE Score distribution')

plt.show()
# TOEFL score

plt.figure(figsize=(8,5))

sns.distplot(df['TOEFL Score'])

plt.title('TOEFL Score distribution')

plt.show()
q = df['TOEFL Score'].quantile(0.01)

df = df[df['TOEFL Score']>q]

df.describe()
# TOEFL score

plt.figure(figsize=(8,5))

sns.distplot(df['TOEFL Score'])

plt.title('TOEFL Score distribution')

plt.show()
# CGPA score

plt.figure(figsize=(8,5))

sns.distplot(df['CGPA'])

plt.title('CGPA Score distribution')

plt.show()
q = df['CGPA'].quantile(0.01)

df = df[df['CGPA']>q]

df.describe()
# CGPA score

plt.figure(figsize=(8,5))

sns.distplot(df['CGPA'])

plt.title('CGPA Score distribution')

plt.show()
# Chance of admit, LOR, SOP

plt.figure(figsize=(8,5))

sns.distplot(df['Chance of Admit '])

plt.title('Chance of Admit distribution')

plt.show()



plt.figure(figsize=(8,5))

sns.distplot(df['LOR '])

plt.title('LOR distribution')

plt.show()



plt.figure(figsize=(8,5))

sns.distplot(df['SOP'])

plt.title('SOP distribution')

plt.show()
q = df['Chance of Admit '].quantile(0.01)

df = df[df['Chance of Admit ']>q]





q = df['LOR '].quantile(0.01)

df = df[df['LOR ']>q]



q = df['SOP'].quantile(0.01)

df = df[df['SOP']>q]



df.describe()
plt.figure(figsize=(8,5))

sns.distplot(df['Chance of Admit '])

plt.title('Chance of Admit distribution')

plt.show()



plt.figure(figsize=(8,5))

sns.distplot(df['LOR '])

plt.title('LOR distribution')

plt.show()



plt.figure(figsize=(8,5))

sns.distplot(df['SOP'])

plt.title('SOP distribution')

plt.show()
print(df['GRE Score'].max())

print(df['TOEFL Score'].max())

print(df['CGPA'].max())

print(df['LOR '].max())

print(df['SOP'].max())

print(df['University Rating'].max())
print(df['Chance of Admit '].max()

     )
df.drop('Serial No.',axis=1, inplace=True)

df.head()
df.dtypes
df.keys()
x = df.loc[:, df.columns != 'Chance of Admit ']

y = df['Chance of Admit ']
print(x.shape)

print(y.shape)
scaler = StandardScaler()

x_trans = scaler.fit_transform(x)
from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test = train_test_split(x_trans, y, test_size=0.15, random_state=3) 
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
lr = LinearRegression()

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
lr.score(x_train, y_train)
lr.score(x_test, y_test)
y_pred
y_test
lr.coef_
df_weights = pd.DataFrame(x.columns.values, columns=['Col_name'])

df_weights['Weight'] = lr.coef_

df_weights.head()
df_pr = pd.DataFrame(y_pred, columns=['Predictions'])

print(df_pr.shape)

df_pr.head()
y_test.keys()
df_pr['Original'] = y_test
df_pr.head()
y_test= y_test.reset_index(drop=True)
df_pr['Original'] = y_test

df_pr.head()
df_pr['difference'] = df_pr['Original'] - df_pr['Predictions']

df_pr.head()
df_pr['Predictions'] = round(df_pr['Predictions'],2)

df_pr['difference'] = round(df_pr['difference'],2)
df_pr.head()
df_pr.shape
lr.predict([[320,110,4,4.5,4.0,8.70,0]])
df.head()