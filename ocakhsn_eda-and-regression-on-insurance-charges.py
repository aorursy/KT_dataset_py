import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/insurance/insurance.csv")

df.head()
df.isnull().sum()
df.shape
df.describe()
df.groupby('region')['charges'].agg(['min', 'max', 'mean'])
plt.figure(figsize=(10, 6))

sns.boxplot(x="region", y="charges", data=df)

plt.title("Box plot of the Regions")

plt.show()
df.groupby('sex')['charges'].agg(['min', 'max', 'mean', 'std'])
plt.figure(figsize=(10, 6))

sns.boxplot(x="sex", y="charges", data=df)

plt.title("Box plot of the Genders")

plt.show()
df.groupby(['region', 'sex'])['charges'].agg(['min', 'max', 'mean', 'std', 'count'])
plt.figure(figsize=(10, 6))

sns.boxplot(x="region", y="charges", hue="sex", data=df)

plt.title("Box plot of the Regions with each gender")

plt.show()
df.groupby('smoker')['charges'].agg(['min', 'max', 'mean', 'std', 'count'])
plt.figure(figsize=(10, 6))

sns.countplot(x="smoker", data=df)

plt.title("Smoker Numbers")

plt.show()
plt.figure(figsize=(10, 6))

sns.boxplot(x="smoker", y="charges", data=df)

plt.title("Box plot of the Smokers")

plt.show()
df.groupby(['region', 'smoker'])['charges'].agg(['min', 'max', 'mean', 'std', 'count'])
plt.figure(figsize=(10, 6))

sns.boxplot(x="region", y="charges", hue="smoker", data=df)

plt.title("Box plot of the Regions")

plt.show()
df['children'].describe()
df['children'].value_counts()
plt.figure(figsize=(10, 6))

sns.distplot(df['children'])

plt.title("Distribution of Children")

plt.show()
df.groupby('children')['charges'].agg(['min', 'max', 'mean', 'std', 'count'])
plt.figure(figsize=(10, 6))

sns.boxplot(x="children", y="charges", data=df)

plt.title("Box plot of the Regions")

plt.show()
plt.figure(figsize=(10, 6))

sns.distplot(df['age'])

plt.title("Distribution of Age")

plt.show()
df.age.describe()
plt.figure(figsize=(10, 6))

sns.scatterplot(x="age", y="charges", data=df)

plt.title("Box plot of the Genders based on Age")

plt.show()
df.groupby(['sex'])['age'].agg(['min', 'max', 'mean', 'count'])
plt.figure(figsize=(10, 6))

sns.boxplot(x="sex", y="age", data=df)

plt.title("Box plot of the Genders based on Age")

plt.show()
df['bmi'].describe()
plt.figure(figsize=(10, 6))

sns.scatterplot(x="bmi", y="charges", data=df)

plt.title("Box plot of the Regions")

plt.show()
df['obesite'] = df['bmi'] > 30

df['obesite'].value_counts()
plt.figure(figsize=(10, 6))

sns.boxplot(x="obesite", y="charges", data=df)

plt.title("Box plot of the Obesite based on charges")

plt.show()
df.groupby('obesite')['charges'].agg(['min', 'max', 'mean', 'count'])
df.groupby(['obesite', 'sex'])['charges'].agg(['min', 'max', 'mean', 'count'])
corr = df.corr()

ax = sns.heatmap(corr, annot=True)
from sklearn.preprocessing import LabelEncoder



columns = ['sex', 'region', 'smoker', 'obesite']



for column in columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])



df.head()
X = df.drop(columns=['charges', 'bmi'])

y = df['charges']



print("X's shape", X.shape)

print("y's shape", y.shape)
from sklearn.preprocessing import scale



columns = X.columns



for column in columns:

    print(column)

    X[column] = scale(X[column])

    

X.head()

    
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



print("There are {} training examples".format(X_train.shape[0]))

print("There are {} test examples".format(X_test.shape[0]))
from sklearn.linear_model import LinearRegression



model = LinearRegression()



model.fit(X_train, y_train)



predictions = model.predict(X_test)
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error





r2 = r2_score(y_test, predictions)

mse = mean_squared_error(y_test, predictions)



print("R2 score is {}".format(r2))

print("Mean Squared Error score is {}".format(mse))
plt.figure(figsize=(10, 6))

sns.scatterplot(x=predictions, y=y_test)

plt.title("Scatter plot of Predictions and Actual Values")

plt.show()