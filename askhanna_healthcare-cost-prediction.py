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
data = pd.read_csv('../input/insurance.csv')
data.head()
data.shape
data.info()
data.describe()
import matplotlib.pyplot as plt
import seaborn as sns
plt.scatter(x=data['age'],y=data['charges'])
plt.xlabel('Age')
plt.ylabel('Charges')
sns.scatterplot(data=data,x='age',y='charges',hue='smoker')
sns.scatterplot(data=data,x='bmi',y='charges',hue='smoker')
plt.figure(figsize=(12,5))
plt.title("Distribution of charges for patients with BMI less than 30")
ax = sns.distplot(data[(data.bmi < 30)]['charges'], color = 'b')
plt.figure(figsize=(12,5))
plt.title("Distribution of charges for patients with BMI less than 30")
ax = sns.distplot(data[(data.bmi >= 30)]['charges'], color = 'b')
g = sns.jointplot(x="bmi", y="charges", data = data,kind="kde", color="r")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")
ax.set_title('Distribution of bmi and charges')
sns.boxplot(data=data,x='region',y='charges')
sns.heatmap(data.corr(),linewidths=0.1,vmax=1.0, 
            square=True,  linecolor='white', annot=True)
sns.countplot(data=data,y='smoker')
sns.countplot(data=data,y='sex')
sns.catplot(x="smoker", kind="count",hue = 'sex', palette="pink", data=data)
sns.countplot(data=data,y='children')

data.head()
new_data = pd.get_dummies(data=data,columns=['sex','smoker','region'])
new_data.head()
X = new_data.drop(columns=['sex_female','smoker_no'])
X.head()
X.drop(columns=['region_southwest'],inplace=True)
y = X['charges']
X.drop(columns=['charges'],inplace = True)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
X_train,X_test,y_train,y_test = train_test_split(X,y)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
model.score(X_test,y_test)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor

quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,y, random_state = 0)

plr = LinearRegression().fit(X_train,Y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print(plr.score(X_test,Y_test))