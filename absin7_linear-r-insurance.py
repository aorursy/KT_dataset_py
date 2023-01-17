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
#Import the important Libraries:
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#Load the dataset and identify the dependent and independent variables:
insurance = pd.read_csv('/kaggle/input/insurance-premium-prediction/insurance.csv')
X = insurance.iloc[: , :-1].values
Y = insurance.iloc[: , 4].values
insurance.head()
#To see the numbers of rows and columns:
insurance.shape
#To check the duplicate values
insurance.duplicated().sum()
#Drop the Duplicate values
insurance = insurance.drop_duplicates()
#Check whether duplicate data is droped or not:
insurance.duplicated().sum()
insurance.age.plot(kind = 'hist')
#Smokers vs Non Smokers
data_smokers = sns.countplot(data = insurance, x = 'smoker')
plt.title("Smokers vs Non Smokers",fontsize = 24)
plt.xlabel("smoker",fontsize = 13)
plt.ylabel("count",fontsize = 13)
plt.figure(figsize= (14,6))
g = sns.countplot(x="age",data=insurance)
g.set_title("Different Age Groups", fontsize=20)
g.set_xlabel("Age", fontsize=15)
g.set_ylabel("Count", fontsize=20)
plt.figure(figsize=(12,6))
g = sns.distplot(insurance["bmi"])
g.set_xlabel("bmi", fontsize=12)
g.set_ylabel("Frequency", fontsize=12)
g.set_title("Frequency Distribuition- bmi", fontsize=20)
insurance.smoker.value_counts().plot(kind = 'pie')
insurance.region.value_counts().plot(kind = "pie")
#We use heatmap to find the correlation between variables:
sns.heatmap(insurance.corr(),vmax = 1,vmin = -1,annot = True)
#To find the unique categories of values:
print(insurance.smoker.unique())
print(insurance.sex.unique())
print(insurance.region.unique())
#Differentiating the Numerical & Categorical values:
all_columns = list(insurance)
numeric_columns = ['age', 'bmi', 'children', 'expenses']
categorical_columns = [x for x in all_columns if x not in numeric_columns]

print("\nNumerical Columns are:")
print(numeric_columns)
print("\nCategorical Columns are:")
print(categorical_columns)

insurance.head()
#To check the percentage of Null values:
percentage_null = (insurance.isnull().sum()/insurance.shape[0])
percentage_null
#ScatterPlot
sns.scatterplot(x = insurance["bmi"], y =insurance["expenses"])

plt.figure(figsize=(10, 7))
sns.scatterplot(x=insurance['bmi'], y=insurance['expenses'],hue=insurance['children'],size=insurance['age'])
plt.figure(figsize=(14,7))
sns.scatterplot(x = insurance['bmi'],y = insurance['expenses'],hue = insurance['region'],size = insurance['bmi'])
plt.figure(figsize=(14,7))
sns.scatterplot(x = insurance['age'],y = insurance['expenses'],hue = insurance['region'],size= insurance['bmi'])
plt.figure(figsize=(14,7))
sns.scatterplot(x = insurance['age'], y = insurance['expenses'],hue = insurance['smoker'],size = insurance['expenses'])
plt.figure(figsize=(14,7))
sns.scatterplot(x = insurance['bmi'], y = insurance['expenses'],hue = insurance['sex'],size  = insurance['age'])
sns.pairplot(insurance,hue ='sex')
sns.pairplot(insurance, hue = 'smoker')
sns.pairplot(insurance, hue = 'region')
plt.figure(figsize=(10,6))
sns.regplot(x = insurance['bmi'], y = insurance['expenses'])
sns.scatterplot(x=insurance['bmi'], y=insurance['expenses'], hue=insurance['smoker'])
sns.lmplot(data = insurance,x = 'bmi',y = 'expenses', hue = 'smoker')
sns.swarmplot(x = insurance['smoker'], y = insurance['expenses'])
g = sns.pairplot(data = insurance)
g.set(xticklabels=[])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
category_col = ['sex','smoker','region']
numerical_columns = [i for i in insurance.columns if i not in category_col]
numeric_columns
#OneHot Encoding
one_hot = pd.get_dummies(insurance[category_col])
new_category_col = pd.concat([insurance[numeric_columns],one_hot],axis =1)
new_category_col.head(10)
#Label Encoding
new_label_category_col = insurance
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in category_col:
    new_label_category_col[i] = label_encoder.fit_transform(new_label_category_col[i])
new_label_category_col.head(10)
#Using OneHotEncoding
X = new_category_col.drop(columns='expenses')
y = insurance['expenses']
train_X,test_X,train_y,test_y = train_test_split(X,y, random_state = 1234,test_size = 0.3)
model = LinearRegression()
model.fit(train_X,train_y)
print("Model_intercept",model.intercept_,"Model_co-efficient",model.coef_)
new_coefficient = pd.DataFrame(data = model.coef_.T,index=X.columns,columns=['coefficients'])
new_coefficient
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

print("Predicting the train data")
train_predict = model.predict(train_X)
print("Predicting the test data")
test_predict = model.predict(test_X)
print("MAE")
print("Train : ",mean_absolute_error(train_y,train_predict))
print("Test  : ",mean_absolute_error(test_y,test_predict))
print("====================================")
print("MSE")
print("Train : ",mean_squared_error(train_y,train_predict))
print("Test  : ",mean_squared_error(test_y,test_predict))
print("====================================")
import numpy as np
print("RMSE")
print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict)))
print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict)))
print("====================================")
print("R^2")
print("Train : ",r2_score(train_y,train_predict))
print("Test  : ",r2_score(test_y,test_predict))
print("MAPE")
print("Train : ",np.mean(np.abs((train_y - train_predict) / train_y)) * 100)
print("Test  : ",np.mean(np.abs((test_y - test_predict) / test_y)) * 100)
#Predicting the Actual vs Predicted Premium
plt.figure(figsize=(14,7))
plt.title("Actual vs Predicted Value",fontsize = 25)
plt.xlabel("Actual Expense")
plt.ylabel("Predicted Expense")
plt.scatter(x = test_y,y = test_predict)