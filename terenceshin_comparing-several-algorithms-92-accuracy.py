# Importing Libraries

import numpy as np

import pandas as pd

import matplotlib as plt

import seaborn as sns

import plotly.express as px

import os
# Reading Data

df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
# Understanding Data

print("Rows, columns: " + str(df.shape))

df.head()
df.describe()
# Missing Values

print(df.isna().sum())
# Histogram of quality

fig = px.histogram(df,x='quality')

fig.show()
# Correlation Matrix

corr = df.corr()

plt.pyplot.subplots(figsize=(15,10))

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
# Create Classification version of target variable

df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
# See proportion of good vs bad wines

df['goodquality'].value_counts()
# Separate feature variables and target variable

X = df.drop(['quality','goodquality'], axis = 1)

y = df['goodquality']
# Normalize feature variables

from sklearn.preprocessing import StandardScaler

X_features = X

X = StandardScaler().fit_transform(X)
# Splitting the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
# Model 1: Decision Tree

from sklearn.metrics import classification_report



from sklearn.tree import DecisionTreeClassifier

model1 = DecisionTreeClassifier(random_state=1)

model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)



print(classification_report(y_test, y_pred1))
# Model 2: Random Forest

from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(random_state=1)

model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)



print(classification_report(y_test, y_pred2))
# Model 3: AdaBoost

from sklearn.ensemble import AdaBoostClassifier

model3 = AdaBoostClassifier(random_state=1)

model3.fit(X_train, y_train)

y_pred3 = model3.predict(X_test)



print(classification_report(y_test, y_pred3))
# Model 4: Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

model4 = GradientBoostingClassifier(random_state=1)

model4.fit(X_train, y_train)

y_pred4 = model4.predict(X_test)



print(classification_report(y_test, y_pred4))
# Model 5: XGBoost

import xgboost as xgb

model5 = xgb.XGBClassifier(random_state=1)

model5.fit(X_train, y_train)

y_pred5 = model5.predict(X_test)



print(classification_report(y_test, y_pred5))
# Feature Importance: Random Forest

feat_importances = pd.Series(model2.feature_importances_, index=X_features.columns)

feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
# Feature Importance: XGBoost

feat_importances = pd.Series(model5.feature_importances_, index=X_features.columns)

feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
df_temp = df[df['goodquality']==1]

df_temp.describe()
df_temp2 = df[df['goodquality']==0]

df_temp2.describe()