%matplotlib inline
import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings(action="ignore") # Filters out all warnings

#warnings.filterwarnings(action="ignore", message="^internal gelsd")
files=os.listdir("../input")
for f in files:

    print(f)
df1=pd.read_csv("../input/"+files[0])

df2=pd.read_csv("../input/"+files[1])

# Using both data files

df = df1.append(df2)
df.head()
df.isna().sum() 

# Since all are 0, means the data contains no Null values so no explicit handling required
df.describe()

#df.columns
df['Serial No.']=range(1,901)

df.index=range(0,900)

df=df.rename(index=str,columns={"Chance of Admit ":"Chance of Admit","LOR ":"LOR"})

df.head(10)
#pd.plotting.scatter_matrix(df,alpha=0.2,figsize=(30,30),diagonal='kde')

#plt.show()

pp = sns.pairplot(data=df,y_vars=['Chance of Admit'],x_vars=['GRE Score', 'TOEFL Score', 'University Rating','SOP','LOR','CGPA','Research'])
df.hist(bins=3,figsize=(10,10))

plt.show()
corr_matrix = df.corr()

corr_matrix["Chance of Admit"].sort_values(ascending=False)
X=df.drop(columns=['Serial No.','Chance of Admit'])

y=df[['Chance of Admit']]
X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



print("Shape of x_train :", X_train.shape)

print("Shape of x_test :", X_test.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_test :", y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
linreg_model = LinearRegression()

linreg_model.fit(X_train, y_train)



y_pred = linreg_model.predict(X_test)



print("---------------------------------------")

print('Coefficients for independent variables:', dict(zip(X.columns,linreg_model.coef_[0])))

print("---------------------------------------")

print('Intercept:', linreg_model.intercept_)

print('Slope:' ,linreg_model.coef_)

print("---------------------------------------")
mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)



# Another detailed way to calculate RMSE below:-



# mse2 = np.sum((linreg_pred - y_test)**2)



# # root mean squared error

# # m is the number of training examples

# rmse = np.sqrt(mse2/len(X_test))



r2 = r2_score(y_test, y_pred)



print("Root Mean Squared Error : ",rmse)

print("R-Square : ", r2)
lr_df = pd.DataFrame({'Actual':np.array(y_test)[:,0],'Predicted':y_pred[:,0]})
sns.regplot(lr_df['Actual'],lr_df['Predicted'],fit_reg=True,color='red')

plt.show()

plt.close()
from sklearn.model_selection import cross_val_score

# cv variable tells in how many parts do we need to divide the data into a stratified manner.

scores = cross_val_score(linreg_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print("RMSE scores :", rmse_scores)

print("Average RMSE value after cross validation :", np.mean(rmse_scores))
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor(random_state=1)

tree_model.fit(X_train, y_train)



y_pred=tree_model.predict(X_test)

tree_model
tree_mse = mean_squared_error(y_test, y_pred)

tree_rmse = np.sqrt(tree_mse)

tree_r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error : ",tree_rmse)

print("R-Square :", tree_r2)
from sklearn.model_selection import cross_val_score

# cv variable tells in how many parts do we need to divide the data into a stratified manner.

scores = cross_val_score(tree_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print("RMSE scores :", rmse_scores)

print("Average RMSE value after cross validation :", np.mean(rmse_scores))
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=10, random_state=2)

forest_model.fit(X_train, y_train)



y_pred=forest_model.predict(X_test)



forest_model
forest_mse = mean_squared_error(y_test, y_pred)

forest_rmse = np.sqrt(forest_mse)

forest_r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error : ",forest_rmse)

print("R-Square :", forest_r2)
from sklearn.model_selection import cross_val_score

# cv variable tells in how many parts do we need to divide the data into a stratified manner.

scores = cross_val_score(forest_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print("RMSE scores :", rmse_scores)

print("Average RMSE value after cross validation :", np.mean(rmse_scores))