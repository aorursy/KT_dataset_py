# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/telcocustomerchurn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.shape
df.isnull().sum()
df.describe()
df['Churn'].value_counts()
sns.countplot(df['Churn'])
#what is the percentage of customer that are leaving
n_r = df[df.Churn == 'No'].shape[0]
n_c = df[df.Churn == 'Yes'].shape[0]

print(round(n_r/(n_r + n_c) * 100, 2), '% stay with company')
print(round(n_c/(n_r + n_c) * 100, 2), '% left with company')
#churn count for both male and female
sns.countplot(x='gender', hue='Churn', data=df)
#churn count for internet service
sns.countplot(x='InternetService', hue='Churn', data=df)
numerical_feature=['tenure', 'MonthlyCharges']
fig, ax = plt.subplots(1, 2, figsize=(28,8))
df[df.Churn=='No'][numerical_feature].hist(bins=20, color='blue', alpha=0.5, ax = ax)
df[df.Churn=='Yes'][numerical_feature].hist(bins=20, color='orange', alpha=0.5, ax = ax)
#remove unnecessery column
clean_df = df.drop('customerID', axis=1)
clean_df.shape
# convert all non numeric to numeric columns
for column in clean_df.columns:
    if clean_df[column].dtype == np.number:
        continue
    print(column)
    clean_df[column] = LabelEncoder().fit_transform(clean_df[column])
# check new data
clean_df.info()
clean_df.head()
#scale the data
X = clean_df.drop(['Churn'], axis=1)
y=clean_df['Churn']

X = StandardScaler().fit_transform(X)
#split the data
x_train,x_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#create model
model = LogisticRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(pred)
print(classification_report(y_test, pred))