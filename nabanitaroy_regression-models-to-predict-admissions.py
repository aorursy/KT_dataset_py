# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

df.head()
#To check for missing values and data types

df.info()
#To ignore serial number

df = df.iloc[:, 1:]

#To check the data distribution except for the serial numbers

df.describe()
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

#Since Research is categorical, correlation plot would not be ideal for that

df_lin = df.drop(["Research"], axis = 1)

#print(df_lin.head())

corr = df_lin.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (10,6))

sns.heatmap(corr, mask=mask, square=True, linewidths=.5, annot = True, cmap = "YlGnBu")
#To select correlations more than 70% and onmly plot them asa function of "Chance of Admit"

df_corr = corr.iloc[:,corr.shape[1] - 1] > .70

feature_list = list(df_corr[df_corr == True].index)

target = feature_list.pop()

feature_list, target
#Visualising CGPA, GRE Score and TOEFL vs Chance of Admit, separating them by "Research"

for i, feature in enumerate(feature_list):

    plt.figure(figsize = (10,10))

    sns.lmplot(x = target, y = feature, data = df, palette = "YlGnBu", fit_reg = True, scatter_kws = {'s': 5}, hue = "Research")

    plt.ylabel(feature)

    plt.xlabel(target)

    plt.show()
#I have seelcted GRE, TOEFL, CGPA and Research to predict chances of admit. To add research to the features list:

feature_list.append("Research")

feature_list
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



X_train1, X_test1, y_train1, y_test1 = train_test_split(df[feature_list], df[target], test_size=0.25, random_state=0)

X_train2, X_test2, y_train2, y_test2 = train_test_split(df.iloc[:, :-1], df[target], test_size=0.25, random_state=0)

scaler = StandardScaler()

X_train1 = scaler.fit_transform(X_train1)

X_test1 = scaler.transform(X_test1)

X_train2 = scaler.fit_transform(X_train2)

X_test2 = scaler.transform(X_test2)
from sklearn.linear_model import LinearRegression

regressor1 = LinearRegression()

regressor2 = LinearRegression()

regressor1.fit(X_train1, y_train1)

y_pred_selected_features = regressor1.predict(X_test1)

regressor2.fit(X_train2, y_train2)

y_pred_all_features = regressor2.predict(X_test2)
from sklearn.metrics import mean_squared_error

error_selected_features = mean_squared_error(y_test1, y_pred_selected_features)

error_all_features = mean_squared_error(y_test2, y_pred_all_features)

error_selected_features, error_all_features, error_selected_features>error_all_features
my_data = np.array([340, 120, 4, 3.5, 3.5, 8.2, 1]).reshape(1, -1)

print(my_data.shape)

#Using Linear Regression

my_data_scaled = scaler.transform(my_data)

my_data_pred = regressor2.predict(my_data_scaled)

#y_pred_me

print(my_data)

print(my_data_scaled)

print(my_data_pred)
from sklearn.ensemble import GradientBoostingRegressor

gbr1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')

gbr2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')

gbr1.fit(X_train1, y_train1)

y_pred_gbr_selected_features = gbr1.predict(X_test1)

gbr2.fit(X_train2, y_train2)

y_pred_gbr_all_features = gbr2.predict(X_test2)
from sklearn.metrics import mean_squared_error

error_gbr_selected_features = mean_squared_error(y_test1, y_pred_gbr_selected_features)

error_gbr_all_features = mean_squared_error(y_test2, y_pred_gbr_all_features)

error_gbr_selected_features, error_gbr_all_features, error_gbr_selected_features>error_gbr_all_features
my_data = np.array([340, 120, 4, 3.5, 3.5, 8.2, 1]).reshape(1, -1)

print(my_data.shape)

#Using Linear Regression

my_data_scaled = scaler.transform(my_data)

my_data_pred = gbr2.predict(my_data_scaled)

#y_pred_me

print(my_data)

print(my_data_scaled)

print(my_data_pred)
feature_importance = gbr2.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())

print(feature_importance)

sorted_idx = np.argsort(feature_importance)

sorted_fi = np.sort(feature_importance)

sorted_features = df.columns[sorted_idx]

print(sorted_fi)

print(sorted_features)

plt.figure(figsize = (20,5))

sns.barplot(sorted_fi, sorted_features, palette = "YlGnBu")

#plt.yticks(feature_importance, sorted_features)

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
#New Splits and Scaling

X_train_cgpa, X_test_cgpa, y_train_cgpa, y_test_cgpa = train_test_split(np.array(df["CGPA"]).reshape(-1,1), df[target], test_size=0.25, random_state=0)

X_train_cgpa = scaler.fit_transform(X_train_cgpa)

X_test_cgpa = scaler.transform(X_test_cgpa)

#Fitting Linear Regression Model

reg_cgpa = LinearRegression()

reg_cgpa.fit(X_train_cgpa, y_train_cgpa)

y_pred_lin_cgpa = reg_cgpa.predict(X_test_cgpa)

error_lin_cgpa = mean_squared_error(y_test_cgpa, y_pred_lin_cgpa)



#Fitting Gradient boosting Regression Model

gbr_cgpa = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')

gbr_cgpa.fit(X_train_cgpa, y_train_cgpa)

y_pred_gbr_cgpa = gbr_cgpa.predict(X_test_cgpa)

error_gbr_cgpa = mean_squared_error(y_test_cgpa, y_pred_gbr_cgpa)

print(error_lin_cgpa, error_gbr_cgpa)
print(X_train2)