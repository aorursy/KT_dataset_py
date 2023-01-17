import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

import seaborn as sns

import xgboost

%matplotlib inline



print(os.listdir("../input/"))

os.chdir("../input/")
df = pd.read_csv('Admission_Predict_Ver1.1.csv')

print(df.head())

df.drop('Serial No.', axis = 1, inplace = True)
print(f'Dataset contains {df.shape[0]} samples, {df.shape[1] - 1} independent features 1 target continuous variable.')
print(df.info())



missing_values = (df.isnull().sum() / len(df)) * 100

print("\nFeatures with missing values: \n", missing_values[missing_values > 0])
df.describe()
sns.heatmap(df.corr(), annot = True)
l = df.columns.values

number_of_columns=df.shape[1]

number_of_rows = len(l)-1/number_of_columns

plt.figure(figsize=(2*number_of_columns,5*number_of_rows))

for i in range(0,len(l)):

    plt.subplot(number_of_rows + 1,number_of_columns,i+1)

    sns.distplot(df[l[i]],kde=True) 
sns.pairplot(df)
X = df[['CGPA', 'GRE Score', 'TOEFL Score']].values

Y = df.iloc[:, -1].values



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
# Linear Regression

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train, y_train)



# XGBoost

xgb_reg = xgboost.XGBRegressor()

xgb_reg.fit(x_train, y_train)



# Random Forest

from sklearn.ensemble import RandomForestRegressor

rand_forest_reg = RandomForestRegressor()

rand_forest_reg.fit(x_train, y_train)
from sklearn.metrics import r2_score

y_pred_lin_reg = reg.predict(x_test)

y_pred_xgb = xgb_reg.predict(x_test)

y_pred_rf = rand_forest_reg.predict(x_test)

print(f"Adjusted R Squared Score for Linear Regression: {r2_score(y_test, y_pred_lin_reg)}")

print(f"Adjusted R Squared Score for XGBoost Regression: {r2_score(y_test, y_pred_xgb)}")

print(f"Adjusted R Squared Score for Random Forest: {r2_score(y_test, y_pred_rf)}")
from sklearn.model_selection import cross_val_score

scores = cross_val_score(reg, x_train, y_train, cv = 6)

print(np.char.center("Linear Regression Score", 40, fillchar = '*'))

print("Scores: ", scores)

print("Accuracy: ", scores.mean() * 100, "%")

print("Standard Deviation: +/-", scores.std(), "\n\n")



print(np.char.center("XGBoost Score", 40, fillchar = '*'))

scores = cross_val_score(xgb_reg, x_train, y_train, cv = 6)

print("Scores: ", scores)

print("Accuracy: ", scores.mean() * 100, "%")

print("Standard Deviation: +/-", scores.std(), "\n\n")



print(np.char.center("Random Forest Score", 40, fillchar = '*'))

scores = cross_val_score(rand_forest_reg, x_train, y_train, cv = 6)

print("Scores: ", scores)

print("Accuracy: ", scores.mean() * 100, "%")

print("Standard Deviation: +/-", scores.std())