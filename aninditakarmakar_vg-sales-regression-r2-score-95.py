## Importing the libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
## Importing the dataset

dataset = pd.read_csv('../input/videogamesales/vgsales.csv')
dataset.info()
dataset.head(10)
# Checking for null values in the dataset

dataset.isnull().values.any()
## Checking which columns contain null values

print(dataset['Rank'].isnull().values.any())

print(dataset['Name'].isnull().values.any())

print(dataset['Platform'].isnull().values.any())

print(dataset['Year'].isnull().values.any())

print(dataset['Genre'].isnull().values.any())

print(dataset['Publisher'].isnull().values.any())

print(dataset['NA_Sales'].isnull().values.any())

print(dataset['EU_Sales'].isnull().values.any())

print(dataset['JP_Sales'].isnull().values.any())

print(dataset['Other_Sales'].isnull().values.any())

print(dataset['Global_Sales'].isnull().values.any())
#Checking the number of missing value rows in the dataset

print(dataset['Year'].isnull().sum())

print(dataset['Publisher'].isnull().sum())
# Removing the missing value rows in the dataset

dataset = dataset.dropna(axis=0, subset=['Year','Publisher'])
dataset.isnull().values.any()
## Defining the features and the dependent variable

x = dataset.iloc[:,1:-1].values

y = dataset.iloc[:,-1].values

print(x[0])

print(y)
## Determining the relevancy of features using heatmap in calculating the outcome variable

corrmat = dataset.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(10,10))

#Plotting heat map

g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,linewidths=.5)

b, t = plt.ylim() # Finding the values for bottom and top

b += 0.5 

t -= 0.5 

plt.ylim(b, t) 

plt.show() 
# Retaining only the useful features of the dataset

# From the heatmap, we can decipher that the columns NA_Sales,JP_Sales,EU_Sales and Other_Sales are the most useful features

# in determining the global sales

x = dataset.iloc[:,6:-1].values

print(x[0])
## Splitting the dataset into independent and dependent vaiables

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)

print(x_test)

print(y_train)

print(y_test)
## Training the multiple linear regression on the training set

from sklearn.linear_model import LinearRegression

regressor_MultiLinear = LinearRegression()

regressor_MultiLinear.fit(x_train,y_train)
## Predicting test results

y_pred = regressor_MultiLinear.predict(x_test)
# Calculating r2 score

from sklearn.metrics import r2_score

r2_MultiLinear = r2_score(y_test,y_pred)

print(r2_MultiLinear)
## Finding out the optimal degree of polynomial regression

from sklearn.preprocessing import PolynomialFeatures

sns.set_style('darkgrid')

scores_list = []

pRange = range(2,6)

for i in pRange :

    poly_reg = PolynomialFeatures(degree=i)

    x_poly = poly_reg.fit_transform(x_train)

    poly_regressor = LinearRegression()

    poly_regressor.fit(x_poly,y_train)

    y_pred = poly_regressor.predict(poly_reg.fit_transform(x_test))

    scores_list.append(r2_score(y_test,y_pred))

plt.plot(pRange,scores_list,linewidth=2)

plt.xlabel('Degree of polynomial')

plt.ylabel('r2 score with varying degrees')

plt.show()
## Training the polynomial regression on the training model

poly_reg = PolynomialFeatures(degree=2)

x_poly = poly_reg.fit_transform(x_train)

poly_regressor = LinearRegression()

poly_regressor.fit(x_poly,y_train)

y_pred = poly_regressor.predict(poly_reg.fit_transform(x_test))

r2_poly = r2_score(y_test,y_pred)

print(r2_poly)
## Finding the optimal number of neighbors for KNN regression

from sklearn.neighbors import KNeighborsRegressor

knnRange = range(1,11,1)

scores_list = []

for i in knnRange:

    regressor_knn = KNeighborsRegressor(n_neighbors=i)

    regressor_knn.fit(x_train,y_train)

    y_pred = regressor_knn.predict(x_test)

    scores_list.append(r2_score(y_test,y_pred))

plt.plot(knnRange,scores_list,linewidth=2,color='green')

plt.xticks(knnRange)

plt.xlabel('No. of neighbors')

plt.ylabel('r2 score of KNN')

plt.show()    
# Training the KNN model on the training set

regressor_knn = KNeighborsRegressor(n_neighbors=7)

regressor_knn.fit(x_train,y_train)

y_pred = regressor_knn.predict(x_test)

r2_knn = r2_score(y_test,y_pred)

print(r2_knn)
# Training the Decision Tree regression on the training model

from sklearn.tree import DecisionTreeRegressor

regressor_Tree = DecisionTreeRegressor(random_state=0)

regressor_Tree.fit(x_train,y_train)
# Predicting test results

y_pred = regressor_Tree.predict(x_test)
# Calculating r2 score

r2_tree = r2_score(y_test,y_pred)

print(r2_tree)
# Finding out the optimal number of trees for Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

forestRange=range(50,500,50)

scores_list=[]

for i in forestRange: 

    regressor_Forest = RandomForestRegressor(n_estimators=i,random_state=0)

    regressor_Forest.fit(x_train,y_train)

    y_pred = regressor_Forest.predict(x_test)

    scores_list.append(r2_score(y_test,y_pred))

plt.plot(forestRange,scores_list,linewidth=2,color='maroon')

plt.xticks(forestRange)

plt.xlabel('No. of trees')

plt.ylabel('r2 score of Random Forest Reg.')

plt.show()    
# Training the Random Forest regression on the training model

regressor_Forest = RandomForestRegressor(n_estimators=100,random_state=0)

regressor_Forest.fit(x_train,y_train)

y_pred = regressor_Forest.predict(x_test)

r2_forest = r2_score(y_test,y_pred)

print(r2_forest)
## Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)

sc_y = StandardScaler()

y_train = sc_y.fit_transform(np.reshape(y_train,(len(y_train),1)))

y_test = sc_y.transform(np.reshape(y_test,(len(y_test),1)))
print(x_train)

print(x_test)

print(y_test)

print(y_train)
## Training the Linear SVR model on the training set

from sklearn.svm import SVR

regressor_SVR = SVR(kernel='linear')

regressor_SVR.fit(x_train,y_train)
## Predicting test results

y_pred = regressor_SVR.predict(x_test)
## Calculating r2 score

r2_linearSVR = r2_score(y_test,y_pred)

print(r2_linearSVR)
## Training the Non-linear SVR model on the training set

from sklearn.svm import SVR

regressor_NonLinearSVR = SVR(kernel='rbf')

regressor_NonLinearSVR.fit(x_train,y_train)
## Predicting test results

y_pred = regressor_NonLinearSVR.predict(x_test)
## Calculating r2 score

r2_NonlinearSVR = r2_score(y_test,y_pred)

print(r2_NonlinearSVR)
## Applying XGBoost Regression model on the training set

from xgboost import XGBRegressor

regressor_xgb = XGBRegressor()

regressor_xgb.fit(x_train,y_train)
## Predicting test results

y_pred = regressor_xgb.predict(x_test)
## Calculating r2 score

r2_xgb = r2_score(y_test,y_pred)

print(r2_xgb)
## Comparing the r2 scores of different models

labelList = ['Multiple Linear Reg.','Polynomial Reg.','K-NearestNeighbors','Decision Tree','Random Forest',

             'Linear SVR','Non-Linear SVR','XGBoost Reg.']

mylist = [r2_MultiLinear,r2_poly,r2_knn,r2_tree,r2_forest,r2_linearSVR,r2_NonlinearSVR,r2_xgb]

for i in range(0,len(mylist)):

    mylist[i]=np.round(mylist[i]*100,decimals=3)

print(mylist)
plt.figure(figsize=(14,8))

ax = sns.barplot(x=labelList,y=mylist)

plt.yticks(np.arange(0, 101, step=10))

plt.title('r2 score comparison among different regression models',fontweight='bold')

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.3f}%'.format(height), (x +0.25, y + height + 0.8))

plt.show()