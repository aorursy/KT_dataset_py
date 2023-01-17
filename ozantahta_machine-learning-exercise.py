# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")
data
data.plot(subplots=True)
data["class"].value_counts()
data.info()
data.columns[data.isnull().any()]
data.describe()
normal = data[data["class"]=="Normal"]
normal
# Linear Regression
y = normal.iloc[:,0].values.reshape(-1,1)
x = normal.iloc[:,1].values.reshape(-1,1)

# import library
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(x,y)

b0 = linear_regression.intercept_
b1 = linear_regression.coef_

y_head = linear_regression.predict(x)

print("Predict 40: ",linear_regression.predict([[40]]))

plt.figure(figsize=(9,9))
plt.scatter(x,y,color = "green")
plt.plot(x,y_head,color = "red")
plt.xlabel("Pelvic Tilt")
plt.ylabel("Pelvic Incidence")
plt.show()
from sklearn.metrics import r2_score
print("r2_score: ",r2_score(y,y_head))
# import dataframe
x1 = normal.loc[:,["pelvic_tilt","lumbar_lordosis_angle"]].values

# Multiple Linear Regression
multiple_linear_reg = LinearRegression()
multiple_linear_reg.fit(x1,y)




print("Multiple Predict (pelvic_incidence) -> 40 and (lumbar_lordosis_angle) -> 26 [pelvic_tilt]:",multiple_linear_reg.predict(np.array([[40,26]])))
first_filter = normal["pelvic_incidence"] > 38
second_filter = normal["pelvic_incidence"] < 41
third_filter = normal["lumbar_lordosis_angle"] < 40
fourth_filter = normal["lumbar_lordosis_angle"] > 10
normal[first_filter & second_filter & third_filter & fourth_filter]
x2 = normal.loc[:,["lumbar_lordosis_angle"]].values

linear_reg2 = LinearRegression()
linear_reg2.fit(x2,y)

y2_head = linear_reg2.predict(x2)

plt.figure(figsize = (9,9))
plt.scatter(x,y,color = "purple",label = "pelvic_incidence")
plt.scatter(x2,y,color = "yellow",label = "lumbar_lordosis_angle")
plt.plot(x,y_head,color = "red",label = "pelvic_incidence_label")
plt.plot(x2,y2_head,color = "red",label = "lumbar_lordosis_angle_label")
plt.legend()
plt.show()
from sklearn.metrics import r2_score
print("r2_score: ",r2_score(y,y2_head))
normal_sort = normal.sort_values(by = "pelvic_tilt")
normal_sort
y = normal_sort.iloc[:,0].values.reshape(-1,1)
x = normal_sort.iloc[:,1].values.reshape(-1,1)


linear_reg3 = LinearRegression()
linear_reg3.fit(x,y)
y_head3 = linear_reg3.predict(x)
plt.figure(figsize = (7,7))
plt.scatter(x,y)
plt.plot(x,y_head3,color="red")

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 3)
x_poly = poly_reg.fit_transform(x)

linear_reg4 = LinearRegression()
linear_reg4.fit(x_poly,y)
y_head4 = linear_reg4.predict(x_poly)
plt.plot(x,y_head4,color = "yellow")
plt.xlabel("Pelvic Tilt")
plt.ylabel("Pelvic Incidence")
print(linear_reg4.predict(poly_reg.fit_transform([[40]])))
from sklearn.metrics import r2_score
print("r2_score: ",r2_score(y,y_head4))
# Decision Tree
y = normal_sort.iloc[:,0].values.reshape(-1,1)
x = normal_sort.iloc[:,1].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

arange = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head5 = tree_reg.predict(arange)

plt.figure(figsize = (9,9))
plt.scatter(x,y,color="blue")
plt.plot(arange,y_head5,color = "red")
plt.xlabel("Pelvic Tilt")
plt.ylabel("Pelvic Incidence")
print(tree_reg.predict([[40]]))
plt.show()

from sklearn.metrics import r2_score
y_head_other = tree_reg.predict(x)
print("r2_score: ",r2_score(y,y_head_other))
# Random Forest
y = normal_sort.iloc[:,0].values.reshape(-1,1)
x = normal_sort.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(n_estimators=300, random_state = 42 ,max_depth= 30)
random_forest.fit(x,y)
print(random_forest.predict([[40]]))
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head6 = random_forest.predict(x_)
plt.figure(figsize=(9,9))
plt.scatter(x,y,color="green")
plt.plot(x_,y_head6,color="red")
plt.show()

from sklearn.metrics import r2_score
y_head_other1 = random_forest.predict(x)
print("r2_score",r2_score(y,y_head_other1))
sns.countplot("class",data=data)
print(data["class"].value_counts())
sns.barplot(data["class"].value_counts().index,data["class"].value_counts().values)
data.info()
data 
color_list = []
for i in data["class"]:
    if i == "Hernia":
        color_list.append("blue")
    elif i == "Normal":
        color_list.append("green")
    else:
        color_list.append("red")
sns.countplot(color_list)
pd.plotting.scatter_matrix(data.iloc[:,data.columns!="class"],
                           figsize=(15,15),
                            marker = "-*-",
                            c = color_list,
                            diagonal = "hist",
                            s = 20,
                            grid = True,
                            alpha = 0.5, edgecolor = "black")
new_data = data.copy()
new_data.sample(5)
new_data["class"] = [1 if i == "Spondylolisthesis" or i == "Hernia" else 0 for i in new_data["class"]]
new_data["class"].value_counts()
x_data = new_data.iloc[:,new_data.columns!="class"].values
y = new_data.loc[:,["class"]].values

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state=1)

from sklearn.neighbors import KNeighborsClassifier
arange = np.arange(1,30)

accuracy_values = []
for i in arange:
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(x_train, y_train)
    accuracy_values.append(KNN.score(x_test,y_test))
    
plt.figure(figsize = (9,9))
plt.plot(arange,accuracy_values,color = "blue",label = "Accuracy")
plt.show()

print("Maximum Accuracy Index: ",accuracy_values.index(max(accuracy_values)))
plt.figure(figsize=(9,9))
plt.scatter(new_data[new_data["class"] == 1]["pelvic_incidence"].values,new_data[new_data["class"] == 1]["pelvic_radius"].values,color = "red",label = "BAD")
plt.scatter(new_data[new_data["class"] == 0]["pelvic_incidence"].values,new_data[new_data["class"] == 0]["pelvic_radius"].values,color = "green",label = "GOOD")
plt.xlabel("pelvic_incidence")
plt.ylabel("pelvic_radius")
plt.legend()
plt.show()

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=accuracy_values.index(max(accuracy_values))) # of course 12
KNN.fit(x,y)
print("Accuracy Score: ", KNN.score(x_test,y_test))
x_ = new_data["pelvic_incidence"].values.reshape(-1,1)
y_ = new_data["pelvic_radius"].values.reshape(-1,1)


from sklearn.ensemble import RandomForestRegressor
random_for = RandomForestRegressor(n_estimators=100,random_state=42)
random_for.fit(x_,y_)
y_head = random_for.predict(x_)

grid = np.arange(min(x_),max(x_),0.01).reshape(-1,1)

plt.figure(figsize=(9,9))
plt.scatter(new_data[new_data["class"] == 1]["pelvic_incidence"].values,new_data[new_data["class"] == 1]["pelvic_radius"].values,color = "red",label = "BAD")
plt.scatter(new_data[new_data["class"] == 0]["pelvic_incidence"].values,new_data[new_data["class"] == 0]["pelvic_radius"].values,color = "green",label = "GOOD")
plt.plot(grid,random_for.predict(grid))
plt.xlabel("pelvic_incidence")
plt.ylabel("pelvic_radius")
plt.legend()
plt.show()

from sklearn.metrics import r2_score
print("R^2 Score: ",r2_score(y_,random_for.predict(x_)))

from sklearn.metrics import mean_squared_error
print('mean_squared_error: ',mean_squared_error(y_,random_for.predict(x_)))
new_data[new_data["class"] == 1]["pelvic_incidence"].values
data.sample(6)
x = new_data["pelvic_incidence"].values.reshape(-1,1)
y = new_data["lumbar_lordosis_angle"].values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x,y)
y_head = linear_reg.predict(x)


plt.figure(figsize = (8,8))
plt.scatter(new_data[new_data["class"] == 1]["pelvic_incidence"],new_data[new_data["class"] == 1]["lumbar_lordosis_angle"],color="red",label = "BAD")
plt.scatter(new_data[new_data["class"] == 0]["pelvic_incidence"],new_data[new_data["class"] == 0]["lumbar_lordosis_angle"],color="green",label = "GOOD")
plt.plot(x,y_head,color="blue",label = "Linear Regression")
plt.xlabel("pelvic_incidence")
plt.ylabel("lumbar_lordosis_angle")
plt.legend()
plt.show()

from sklearn.metrics import r2_score
print(r2_score(y,y_head))
new_data.sample(5)
# Cross Validation
from sklearn.model_selection import cross_val_score
reg = LinearRegression()
k = 5
cv_result = cross_val_score(reg,x,y,cv = k)
print("CV Scores",cv_result)
print("CV Scores Average: ",np.mean(cv_result))
# Ridge
from sklearn.linear_model import Ridge
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2, test_size = 0.3)
ridge = Ridge(alpha = 0.1, normalize = True)
ridge.fit(x_train,y_train)
ridge_predict = ridge.predict(x_test)
print('Ridge score: ',ridge.score(x_test,y_test))
plt.scatter(x_test,y_test)
plt.plot(x_test,ridge.predict(x_test),color="red")
print(ridge.intercept_)
print(ridge.coef_)
sse_values = []
slope = np.linspace(-0.2,2,len(y_test))
intercept = 11.29623421
for i in range(len(y_test)):
    sse = (y_test.ravel()[i] - (intercept + slope[i]* x_test.ravel()[i]))
    sse_values.append(sse**2)
sse_values1 = []
slope = np.linspace(-0.2,2,len(y_test))
intercept = 11.29623421
for i in range(len(y_test)):
    sse = (y_test.ravel()[i] - (intercept + slope[i]* x_test.ravel()[i]))**2 + (100 + (slope[i])**2)
    sse_values1.append(sse)
sse_values2 = []
slope = np.linspace(-0.2,2,len(y_test))
intercept = 11.29623421
for i in range(len(y_test)):
    sse = (y_test.ravel()[i] - (intercept + slope[i]* x_test.ravel()[i]))**2 + (500 + (slope[i])**2)
    sse_values2.append(sse)
sse_values3 = []
slope = np.linspace(-0.2,2,len(y_test))
intercept = 11.29623421
for i in range(len(y_test)):
    sse = (y_test.ravel()[i] - (intercept + slope[i]* x_test.ravel()[i]))**2 + (1000 + (slope[i])**2)
    sse_values3.append(sse)
sse_values4 = []
slope = np.linspace(-0.2,2,len(y_test))
intercept = 11.29623421
for i in range(len(y_test)):
    sse = (y_test.ravel()[i] - (intercept + slope[i]* x_test.ravel()[i]))**2 + (2000 + (slope[i])**2)
    sse_values4.append(sse)
len(sse_values4)
for i in range(len(y_test)):
    print(slope[i],sse_values[i])
aralık = np.arange(0,100,10)
new1 = []
other = []
other1 = []
other2 = []
other3 = []
other4 = []
for i in aralık:
    new1.append(slope[i])
    other.append(sse_values[i])
    other1.append(sse_values1[i])
    other2.append(sse_values2[i])
    other3.append(sse_values3[i])
    other4.append(sse_values4[i])
plt.figure(figsize = (9,9))
plt.plot(new1,other,color="green",label="lambda: 0")
plt.plot(new1,other1,color="red",label="lambda: 1")
plt.plot(new1,other2,color="blue",label="lambda: 5")
plt.plot(new1,other3,color="orange",label="lambda: 10")
plt.plot(new1,other4,color="yellow",label="lambda: 20")
plt.legend()
plt.show()
