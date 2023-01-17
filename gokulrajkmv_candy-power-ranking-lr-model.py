import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

import plotly as ply
import cufflinks as cf
df = pd.read_csv('../input/the-ultimate-halloween-candy-power-ranking/candy-data.csv')
df.info()
df.head()
df.describe()
# Exploratory Data Analysis

df_cor = df.corr()

plt.figure(figsize=(12,6))
sns.heatmap(df_cor,cmap='YlGn',annot=True)
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.set_context('notebook',font_scale=1)
sns.distplot(df['winpercent'],color='#16A085',bins=6)
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.set_context('notebook',font_scale=1)
sns.barplot(x='chocolate',y='sugarpercent',data=df,hue='caramel',palette='YlOrBr')
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.set_context('notebook',font_scale=1)
sns.barplot(x='chocolate',y='winpercent',palette='summer_r',data=df)

df_pplot = df[['sugarpercent','pricepercent','winpercent']] 

sns.pairplot(df_pplot)
# Logistic Regression model

from sklearn.model_selection import train_test_split
X = df[['fruity', 'caramel', 'peanutyalmondy', 'nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus', 'sugarpercent', 'pricepercent', 'winpercent']]

y = df[['chocolate']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Train and fit a logistic regression model

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter=10000)
# fit a logistic regression model

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
# confusion matrixs

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)