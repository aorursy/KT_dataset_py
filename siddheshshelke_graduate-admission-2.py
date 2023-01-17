import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import random
admissions = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")

admissions = admissions.drop('Serial No.',axis = 1)
admissions.head()
admissions.describe()
# Basic correlogram

sns.pairplot(admissions)
corr = admissions.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV,train_test_split

from sklearn.metrics import mean_absolute_error
X = admissions.drop('Chance of Admit ',axis = 1)

y = admissions['Chance of Admit ']



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = .25,random_state = 123)
lin_model = LinearRegression()
lin_model.fit(X_train,y_train)
print('Mean absolute error for linear model: %0.4f' %mean_absolute_error(y_val,lin_model.predict(X_val)))
rf_model = RandomForestRegressor(n_estimators = 100,random_state = 42)

rf_model.fit(X_train,y_train)
print('Mean absolute error for linear model: %0.4f' %mean_absolute_error(y_val,rf_model.predict(X_val)))
feature_importance = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(10, 6))

sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))

plt.title('Random Forest Feature Importance')

plt.tight_layout()