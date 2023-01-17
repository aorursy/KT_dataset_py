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
dataset=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
print(dataset.head())
# to check skewness of GRE Score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew 
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the distribution 
sns.distplot(dataset['GRE Score'], color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="GRE Score")
ax.set(title="GRE Score distribution")
sns.despine(trim=True, left=True)
plt.show()


print("skew value: ", skew(dataset['GRE Score']))
# to check skewness of TOEFL Score
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the distribution 
sns.distplot(dataset['TOEFL Score'], color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="TOEFL Score")
ax.set(title="TOEFL Score distribution")
sns.despine(trim=True, left=True)
plt.show()

print("skew value: ", skew(dataset['TOEFL Score']))
# to check skewness of CGPA
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the distribution 
sns.distplot(dataset['CGPA'], color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="CGPA")
ax.set(title="CGPA distribution")
sns.despine(trim=True, left=True)
plt.show()

print("skew value: ", skew(dataset['CGPA']))
# to check skewness of target variable
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the distribution 
sns.distplot(dataset['Chance of Admit '], color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="Chance of Admit")
ax.set(title="Chance of Admit distribution")
sns.despine(trim=True, left=True)
plt.show()

print("skew value: ", skew(dataset['Chance of Admit ']))
# we need to normalize the target variable
from scipy import stats

crim_box=stats.boxcox(dataset['Chance of Admit '])[0]
print(skew(crim_box))


# to check skewness of target variable
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(crim_box, color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="Chance of Admit")
ax.set(title="Chance of Admit distribution")
sns.despine(trim=True, left=True)
plt.show()
dataset['Chance of Admit ']=crim_box
print("skew value: ", skew(dataset['Chance of Admit ']))
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

y=dataset[dataset.columns[8]].values;
X=dataset[dataset.columns[[1,2,3,4,5,6,7]]].values

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
model=RandomForestRegressor()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("Accuracy: ", model.score(X_test,y_test))


#print(y_pred)
#print(y_test)
#print(accuracy_score(y_pred,y_test))

print("MSE: ", mean_squared_error(y_pred,y_test))
print("RMSE: ", np.sqrt(mean_squared_error(y_pred,y_test)))
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("Accuracy: ", model.score(X_test,y_test))


#print(y_pred)
#print(y_test)
#print(accuracy_score(y_pred,y_test))

print("MSE: ", mean_squared_error(y_pred,y_test))
print("RMSE: ", np.sqrt(mean_squared_error(y_pred,y_test)))
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
model=xgb.XGBRegressor()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("Accuracy: ", model.score(X_test,y_test))


#print(y_pred)
#print(y_test)
#print(accuracy_score(y_pred,y_test))

print("MSE: ", mean_squared_error(y_pred,y_test))
print("RMSE: ", np.sqrt(mean_squared_error(y_pred,y_test)))