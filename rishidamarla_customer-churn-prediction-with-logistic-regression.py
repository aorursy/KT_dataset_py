import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.info()
df.describe()
df.isnull().any()
df.shape
df['Churn'].value_counts()
sns.countplot(df['Churn'])
sns.countplot(x='gender', hue='Churn', data = df)
sns.countplot(x='InternetService', hue = 'Churn', data = df)
numerical_features = ['tenure', 'MonthlyCharges']

fig, ax = plt.subplots(1, 2, figsize = (28,8))

df[df.Churn == 'No'][numerical_features].hist(bins=20, color='blue', alpha=0.5, ax = ax)

df[df.Churn == 'Yes'][numerical_features].hist(bins=20, color='orange', alpha=0.5, ax = ax)
df2 = df.drop('customerID', axis=1)
from sklearn.preprocessing import LabelEncoder

for column in df2:

    if df2[column].dtype == np.number:

        continue

    df2[column] = LabelEncoder().fit_transform(df2[column])
df2.dtypes
df2.head()
X = df2.drop('Churn', axis=1)

y = df2['Churn']
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))