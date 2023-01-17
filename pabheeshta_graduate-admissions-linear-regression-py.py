import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.style as style

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split 

from sklearn import preprocessing

from sklearn import metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor

style.use(['fivethirtyeight', 'seaborn-whitegrid'])
grad_DF = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
grad_DF.head()
grad_DF.shape
list(grad_DF.columns)
grad_DF.rename(columns = {"LOR ": "LOR", "Chance of Admit ": "Chance of Admit"}, inplace = True)
list(grad_DF.columns)
grad_DF.drop(columns=['Serial No.'], inplace = True)
grad_DF.head()
f, ax1 = plt.subplots(4, 2, figsize = (13,17))

sns.countplot(data = grad_DF, x = "University Rating", ax = ax1[0,0])

sns.countplot(data = grad_DF, x = "Research", ax = ax1[0,1])

sns.distplot(grad_DF["GRE Score"], ax = ax1[1, 0])

sns.distplot(grad_DF["TOEFL Score"], ax = ax1[1, 1])

sns.countplot(data = grad_DF, x = "SOP", ax = ax1[2,0])

sns.countplot(data = grad_DF, x = "LOR", ax = ax1[2,1])

sns.distplot(grad_DF["CGPA"], ax = ax1[3, 0])

sns.distplot(grad_DF["Chance of Admit"], ax = ax1[3, 1])

plt.show()
grad_DF[['GRE Score', 'TOEFL Score', 'CGPA', 'Chance of Admit']].describe()
grad_corr = grad_DF.corr()

plt.figure(figsize = (7,7))

sns.heatmap(grad_corr, annot = True, linewidths = 1.2, linecolor = 'white')

plt.show()
X = grad_DF[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]

Y = grad_DF['Chance of Admit']
X_scaled = preprocessing.scale(X)

X_scaled_DF = pd.DataFrame(data = X_scaled, columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'])

X_scaled_DF.head()
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
grad_reg = LinearRegression()

grad_reg.fit(X_train, Y_train)
grad_pred = grad_reg.predict(X_test)
vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X_scaled_DF.values, i) for i in range(X_scaled_DF.shape[1])]

vif["Features"] = X_scaled_DF.columns



vif
pred_DF = pd.DataFrame({'Actual': Y_test, 'Predicted': grad_pred})

pred_DF.head(20)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, grad_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(Y_test, grad_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, grad_pred)))
metrics.r2_score(Y_test, grad_pred)