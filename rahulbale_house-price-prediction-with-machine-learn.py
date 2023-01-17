# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

#model selection and evaluation
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,mean_squared_error


# Model libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
data.head()
data["date"] = pd.to_datetime(data["date"])
data["date"]
data.info()
data.describe()
data.isnull().sum()
data.date=pd.to_datetime(data.date)
data.date=pd.to_numeric(data.date)
data.corr()
data.columns
plt.figure(figsize=(20, 16))
plt.title('Pearson Correlation Matrix',fontsize=25)
sns.heatmap(data.corr(),linewidths=0.25,square=True,cmap="BuGn",linecolor='w',annot=True);
data.columns
fig, ax= plt.subplots(figsize=(27,30), ncols=3, nrows=6)

sns.scatterplot(x='bedrooms', y='price', data=data, ax=ax[0][0])
sns.scatterplot(x='bathrooms', y='price', data=data, ax=ax[0][1])
sns.scatterplot(x='sqft_living', y='price', data=data, ax=ax[0][2])
sns.scatterplot(x='sqft_lot', y='price', data=data, ax=ax[1][0])
sns.scatterplot(x='floors', y='price', data=data, ax=ax[1][1])
sns.scatterplot(x='waterfront', y='price', data=data, ax=ax[1][2])
sns.scatterplot(x='view', y='price', data=data, ax=ax[2][0])
sns.scatterplot(x='condition', y='price', data=data, ax=ax[2][1])
sns.scatterplot(x='grade', y='price', data=data, ax=ax[2][2])
sns.scatterplot(x='sqft_above', y='price', data=data, ax=ax[3][0])
sns.scatterplot(x='sqft_basement', y='price', data=data, ax=ax[3][1])
sns.scatterplot(x='yr_built', y='price', data=data, ax=ax[3][2])
sns.scatterplot(x='yr_renovated', y='price', data=data, ax=ax[4][0])
sns.scatterplot(x='zipcode', y='price', data=data, ax=ax[4][1])
sns.scatterplot(x='lat', y='price', data=data, ax=ax[4][2])
sns.scatterplot(x='long', y='price', data=data, ax=ax[5][0])
sns.scatterplot(x='sqft_living15', y='price', data=data, ax=ax[5][1])
sns.scatterplot(x='sqft_lot15', y='price', data=data, ax=ax[5][2])

model_name= [] #Modal name 
MSE_score= []  #Mean squared error
R2_score=[]    #R^2 Score
space=data['sqft_living']
price=data['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print("X Train Shape", x_train.shape)
print("Y Train Shape", y_train.shape)
print("X Test Shape", x_test.shape)
print("Y Test Shape", y_test.shape)

sim_lin=LinearRegression()
sim_lin.fit(x_train,y_train)
sim_lin_pred=sim_lin.predict(x_test)
print("Coefficient of Simple Lin Regression : ",sim_lin.coef_[:])
print("Intercept of Simple Lin Regression : ",sim_lin.intercept_)
model_name.append("Simple Linear Regression")
MSE_score.append(mean_squared_error(y_test,sim_lin_pred))
R2_score.append(r2_score(y_test,sim_lin_pred))

print('Mean Squared Error',mean_squared_error(y_test,sim_lin_pred))
print('R score %0.2f'%r2_score(y_test,sim_lin_pred))

x=data.drop(['id', 'date', 'price','zipcode'],axis=1)
y=data['price']

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=0)
print("X Train Shape", x_train.shape)
print("Y Train Shape", y_train.shape)
print("X Test Shape", x_test.shape)
print("Y Test Shape", y_test.shape)
multi_reg=LinearRegression()
multi_reg.fit(xtrain,ytrain)
multi_pred=multi_reg.predict(xtest)

model_name.append('Multi Linear Regression')
MSE_score.append(mean_squared_error(ytest,multi_pred))
R2_score.append(r2_score(ytest,multi_pred))
print("Coefficient of Multi Lin Regression : ",multi_reg.coef_[0:5])
print("Intercept of Multi Lin Regression : ",multi_reg.intercept_)
print('Mean Squared Error',mean_squared_error(ytest,multi_pred))
print('R score %0.2f'%r2_score(ytest,multi_pred))

ridge=Ridge()
ridge.fit(xtrain,ytrain)
ridge_pred=ridge.predict(xtest)

model_name.append('Ridge Regression')
MSE_score.append(mean_squared_error(ytest,ridge_pred))
R2_score.append(r2_score(ytest,ridge_pred))
print("Coefficient of Ridge Regression : ",ridge.coef_[0:5])
print("Intercept of Ridge Regression : ",ridge.intercept_)
print('Mean Squared Error',mean_squared_error(ytest,ridge_pred))
print('R score %0.2f'%r2_score(ytest,ridge_pred))
lasso_model = Lasso()
lasso_model.fit(xtrain,ytrain)
lasso_model_predict = lasso_model.predict(xtest)

model_name.append('Lasso Regression')
MSE_score.append(mean_squared_error(ytest,lasso_model_predict))
R2_score.append(r2_score(ytest,lasso_model_predict))
print("Coefficient of Lasso Regression : ",lasso_model.coef_[0:5])
print("Intercept of Lasso Regression : ",lasso_model.intercept_)
print('Mean Squared Error',mean_squared_error(ytest,lasso_model_predict))
print('R score %0.2f'%r2_score(ytest,lasso_model_predict))
decision_tree = DecisionTreeRegressor()
decision_tree.fit(xtrain,ytrain)
decision_tree_predict = decision_tree.predict(xtest)

model_name.append('Decision Tree Regression')
MSE_score.append(mean_squared_error(ytest,decision_tree_predict))
R2_score.append(r2_score(ytest,decision_tree_predict))
print('Mean Squared Error',mean_squared_error(ytest,decision_tree_predict))
print('R score %0.2f'%r2_score(ytest,decision_tree_predict))
plt.subplots(figsize=(15, 5))
sns.barplot(x=R2_score,y=model_name,palette = sns.cubehelix_palette(len(R2_score)))
plt.xlabel("Score")
plt.ylabel("Regression Modal")
plt.title('Regression Scores')
plt.show()
#set ids as Id and predict survival 
ids = data['id']
predict = decision_tree.predict(data.drop(["id","date","price","zipcode"],axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'HouseID' : ids, 'Price': predict})
output.to_csv('submission.csv', index=False)