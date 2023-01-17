import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')

df.head()
df.isna().sum()
df.drop('Serial No.', axis=1, inplace=True)

df.head()
import matplotlib.pyplot as plt

import seaborn as sns



fig = sns.distplot(df['GRE Score'])

plt.title('GRE Score')

plt.show()



fig = sns.distplot(df['TOEFL Score'])

plt.title('TOEFL Score')

plt.show()



fig = sns.distplot(df['University Rating'])

plt.title('University Rating')

plt.show()



fig = sns.distplot(df['CGPA'])

plt.title('CGPA')

plt.show()
sns.regplot(x = 'GRE Score', y='TOEFL Score', data=df)

plt.title('GRE Score vs TOEFL Score')

plt.show()
sns.regplot(x = 'GRE Score', y='CGPA', data=df)

plt.title('GRE Score vs CGPA')

plt.show()
sns.regplot(x = 'CGPA', y='TOEFL Score', data=df)

plt.title('CGPA vs TOEFL Score')

plt.show()
df.describe().plot(kind='area', figsize=(20,8), fontsize=25, table=True)

plt.xlabel('Statistics')

plt.ylabel('Value')

plt.title('Statistics for Admissions')
plt.figure(1, figsize=(10,6))

plt.subplot(1, 4, 1)

plt.boxplot(df['GRE Score'])

plt.title('GRE Score')



plt.subplot(1, 4, 2)

plt.boxplot(df['TOEFL Score'])

plt.title('TOEFL Score')



plt.subplot(1, 4, 3)

plt.boxplot(df['University Rating'])

plt.title('University Rating')



plt.subplot(1, 4, 4)

plt.boxplot(df['CGPA'])

plt.title('CGPA')



plt.show()
fig = plt.gcf() # Creates a new figure if no current exists

fig.set_size_inches(10,10)

dropSelf = np.zeros_like(df.corr())  # Extracts value with same shape

dropSelf[np.triu_indices_from(dropSelf)] = True # Extracts upper indices

colormap = sns.diverging_palette(220, 10, as_cmap=True)

fig = sns.heatmap(df.corr(), annot=True, square=True, cmap = colormap, mask= dropSelf)
sns.scatterplot(x='GRE Score', y='TOEFL Score', hue='Research', data=df)
df['Research'].value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, explode=[0.1,0])
sns.pairplot(df)
X = df.iloc[:, :-1].values

Y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



models= [

    ['Linear Regression: ',LinearRegression()],

    ['SVR:', SVR()],

    ['Decision Tree Regression:', DecisionTreeRegressor()],

    ['Random Forest Refression:', RandomForestRegressor()]

]



for name, model in models:

    model = model

    model.fit(X_train, Y_train)

    prediction = model.predict(X_test)

    print(name, mean_squared_error(Y_test, prediction))