import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn import metrics
style.use('fivethirtyeight')
medical_DF = pd.read_csv("../input/insurance/insurance.csv")
medical_DF.head(10)
medical_DF.shape
medical_DF.isnull().sum()
f, ax1 = plt.subplots(2, 2, figsize = (10,10))
sns.countplot(data = medical_DF, x = "sex", ax = ax1[0,0])
sns.countplot(data = medical_DF, x = "children", ax = ax1[0,1])
sns.countplot(data = medical_DF, x = "smoker", ax = ax1[1,0])
sns.countplot(data = medical_DF, x = "region", ax = ax1[1,1])
plt.show()
f, ax2 = plt.subplots(2, 2, figsize = (12,10))
sns.violinplot(data = medical_DF, x = "age", ax = ax2[0,0], color = '#d43131')
sns.violinplot(data = medical_DF, x = "bmi", ax = ax2[0,1], color = '#3152d2')
sns.violinplot(data = medical_DF, x = "charges", ax = ax2[1,0], color = '#bd7a17')
f.delaxes(ax = ax2[1,1])
plt.show()
medical_DF[['age', 'bmi', 'charges']].describe()
plt.figure(figsize = (10,5))
sns.violinplot(data = medical_DF, x = "charges", y = "sex", hue = "sex")
plt.show()
medical_DF.groupby(['sex']).describe()['charges']
plt.figure(figsize = (10,5))
sns.violinplot(data = medical_DF, x = "charges", y = "smoker", hue = "smoker")
plt.show()
medical_DF.groupby(['smoker']).describe()['charges']
plt.figure(figsize = (10,10))
sns.violinplot(data = medical_DF, x = "charges", y = "region", hue = "region")
plt.show()
medical_DF.groupby(['region']).describe()['charges']
plt.figure(figsize = (10,10))
sns.violinplot(data = medical_DF, x = "charges", y = "children", hue = "children")
plt.show()
medical_DF.groupby(['children']).describe()['charges']
plt.figure(figsize = (6,6))
sns.scatterplot(data = medical_DF, x = "charges", y = "bmi")
plt.show()
plt.figure(figsize = (6,6))
sns.scatterplot(data = medical_DF, x = "charges", y = "age")
plt.show()
med_corr = medical_DF.corr()
plt.figure(figsize = (7,7))
sns.heatmap(med_corr, annot = True, linewidths = 1.2, linecolor = 'white')
plt.show()
medical_DF_1 = medical_DF
medical_DF_1 = pd.get_dummies(medical_DF, columns = ['region', 'children', 'sex'])
medical_DF_1.drop(columns = ['sex_female'], inplace = True)
medical_DF_1["smoker"].replace({"yes": 1, "no": 0}, inplace=True)
medical_DF_1.head(10)
med_corr = medical_DF_1.corr()
plt.figure(figsize = (14,10))
sns.heatmap(med_corr, annot = True, linewidths = 1.2, linecolor = 'white')
plt.show()
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

medical_DF_1["age"] = NormalizeData(medical_DF_1["age"])
medical_DF_1["bmi"] = NormalizeData(medical_DF_1["bmi"])

medical_DF_1.head(10)
X = medical_DF_1.drop(columns=['charges'])
Y = medical_DF_1['charges']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
med_reg = LinearRegression()
med_reg.fit(X_train, Y_train)
med_pred = med_reg.predict(X_test)
med_pred_DF = pd.DataFrame({'Actual': Y_test, 'Predicted': med_pred})
med_pred_DF.head(20)
metrics.r2_score(Y_test, med_pred)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, med_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, med_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, med_pred)))
X_poly = PolynomialFeatures(degree = 2, include_bias = False).fit_transform(X)
X_poly_train, X_poly_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=42)
med_poly_reg = LinearRegression().fit(X_poly_train, Y_train)
med_poly_pred = med_poly_reg.predict(X_poly_test)
metrics.r2_score(Y_test, med_poly_pred)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, med_poly_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, med_poly_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, med_poly_pred)))