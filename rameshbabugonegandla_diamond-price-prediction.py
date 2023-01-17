from IPython.display import Image
import os
Image("../input/diamondimages/Introduction.PNG")
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
# Loading Libraries
import pandas as pd # for data analysis
import numpy as np # for scientific calculation
import seaborn as sns # for statistical plotting
import matplotlib.pyplot as plt # for plotting
%matplotlib inline
#Reading Diamond Price Prediction data set.
import os
for dirname, _, filenames in os.walk('/kaggle/input/diamond/diamonds_prediction.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
diamondprediction_eda=pd.read_csv('/kaggle/input/diamond/diamonds_prediction.csv')
diamondprediction_eda.describe()
print(diamondprediction_eda.shape)
print(diamondprediction_eda.head(2))
print(diamondprediction_eda.info())
# Identify Duplicate Records
duplicate_records = diamondprediction_eda[diamondprediction_eda.duplicated()]
print("Duplicate Rows except first occurrence based on all columns are :")
print(len(duplicate_records))
print(diamondprediction_eda.shape)
print(duplicate_records.head(2))
# Missing Data Percentage
total = diamondprediction_eda.isnull().sum().sort_values(ascending=False)
percent = (diamondprediction_eda.isnull().sum()/diamondprediction_eda.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of Missing Values', fontsize=15)
plt.title('Percentage of Missing Data by Feature', fontsize=15)
missing_data.head()
diamondprediction_eda.isnull().sum()
# Analysis of Non-numerical columns
diamondprediction_eda.describe(include=['O'])
# Cut Types
# cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
print(diamondprediction_eda['cut'].unique())
print("Type of Cuts: %s" % (diamondprediction_eda['cut'].nunique()))
print(diamondprediction_eda['cut'].value_counts())
# Color Types
# color diamond colour, from J (worst) to D (best)
print(diamondprediction_eda['color'].unique())
print("Type of Color: %s" % (diamondprediction_eda['color'].nunique()))
print(diamondprediction_eda['color'].value_counts())
# Clarity Types
# clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
print(diamondprediction_eda['clarity'].unique())
print("Type of Clarity: %s" % (diamondprediction_eda['clarity'].nunique()))
print(diamondprediction_eda['clarity'].value_counts())
# Remove unwanted column
print(diamondprediction_eda.head(2))
diamondprediction_eda = diamondprediction_eda.drop(['Unnamed: 0'], axis =1)
print(diamondprediction_eda.head(2))
# Count plot for categorical variables.
plt.rcParams['figure.figsize'] = (10.0, 5.0)
plt.rcParams['font.family'] = "serif"
fig, ax =plt.subplots(2,2)
sns.countplot(diamondprediction_eda['cut'], ax=ax[0,0])
sns.countplot(diamondprediction_eda['color'], ax=ax[0,1])
sns.countplot(diamondprediction_eda['cut'], ax=ax[1,0])
sns.countplot(diamondprediction_eda['clarity'], ax=ax[1,1])
fig.show();
# Mean Encoding technique has been choosen to ensure better encoding amoung other technique's based on the above plots.
# Verified the top 2 sample data after mean-encoding technique.
mean_encode_cut = diamondprediction_eda.groupby('cut')['price'].mean()
mean_encode_color = diamondprediction_eda.groupby('color')['price'].mean()
mean_encode_clarity = diamondprediction_eda.groupby('clarity')['price'].mean()
diamondprediction_eda.loc[:,'cut_mean_enc']=diamondprediction_eda['cut'].map(mean_encode_cut)
diamondprediction_eda.loc[:,'color_mean_enc']=diamondprediction_eda['color'].map(mean_encode_color)
diamondprediction_eda.loc[:,'clarity_mean_enc']=diamondprediction_eda['clarity'].map(mean_encode_clarity)
diamondprediction_eda.head(2)
# Drop categorical variables
diamondprediction_eda=diamondprediction_eda.drop(['cut','color','clarity'],axis=1)
diamondprediction_eda.rename(columns = {'cut_mean_enc':'cut'}, inplace = True)
diamondprediction_eda.rename(columns = {'color_mean_enc':'color'}, inplace = True)
diamondprediction_eda.rename(columns = {'clarity_mean_enc':'clarity'}, inplace = True)
diamondprediction_eda.head(2)
diamondprediction_eda.describe()
# Removing (statistical) outliers for dataset
print(diamondprediction_eda.shape)
Q1 = diamondprediction_eda.quantile(0.25) #(0.19485)
#print(Q1)
Q3 = diamondprediction_eda.quantile(0.75) #(0.80515)
#print(Q3)
IQR = Q3 - Q1
#print(IQR)
diamondprediction_eda_outliers = diamondprediction_eda[~((diamondprediction_eda < (Q1 - 1.5 * IQR)) |(diamondprediction_eda > (Q3 + 1.5 * IQR))).any(axis=1)]
print(diamondprediction_eda_outliers.shape)
# Distribution plot for output variable.
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,8))
sns.distplot(diamondprediction_eda.price,ax=ax[0],kde=False)
sns.distplot(diamondprediction_eda_outliers.price,ax=ax[1],kde=False)
plt.show()
plt.tight_layout();
plt.boxplot(diamondprediction_eda['price'])
plt.show()
plt.boxplot(diamondprediction_eda_outliers['price'])
plt.show();
diamondprediction_eda_outliers.head(2)
# Generated HeatMap after removal of outliers and you can see the difference clearly here.
corr = diamondprediction_eda_outliers.corr()
ax = sns.heatmap( corr,vmin=-1, vmax=1, center=0,  cmap=sns.diverging_palette(20, 220, n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');
X=diamondprediction_eda_outliers.drop(['price'],axis=1)
Y=diamondprediction_eda_outliers['price']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

R2_Accuracy = []
models = ['Linear Regression', 'Lasso Regression','Ridge Regression','ElasticNet Regression','DecisionTree Regression','RandomForest Regression','KNeighbours Regression','AdaBoost Regression']
linear_reg = LinearRegression()
linear_reg.fit(X_train , Y_train)
cross_validation_accuracy = cross_val_score(estimator = linear_reg, X = X_train, y = Y_train, cv = 5,verbose = 1)
y_pred = linear_reg.predict(X_test)
print('\nLinear Regression')
print('Accuracy Score : %.6f' % linear_reg.score(X_test, Y_test))

print('\nCross Validation Accuracy')
print(cross_validation_accuracy)

rmse = mean_squared_error(Y_test, y_pred)**0.5
r2 = r2_score(Y_test, y_pred)

print('\nMetrics')
print('RMSE   : %0.6f ' % rmse)
print('R2     : %0.6f ' % r2)

R2_Accuracy.append(r2);

# Putting together the coefficient and their corrsponding variable names  
linear_reg_coefficient = pd.DataFrame() 
linear_reg_coefficient["Columns"] = X_train.columns 
linear_reg_coefficient['Coefficient Estimate'] = pd.Series(linear_reg.coef_) 
print(linear_reg_coefficient) 

# Let’s plot a bar chart of above coefficients using matplotlib plotting library.
# plotting the coefficient score 
fig, ax = plt.subplots(figsize =(10, 5)) 
  
color =['tab:gray', 'tab:blue', 'tab:orange',  
'tab:green', 'tab:red', 'tab:purple', 'tab:brown',  
'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',  
'tab:orange', 'tab:green', 'tab:blue', 'tab:olive'] 
  
ax.bar(linear_reg_coefficient["Columns"],  
linear_reg_coefficient['Coefficient Estimate'],color = color) 
  
ax.spines['bottom'].set_position('zero') 
  
plt.style.use('ggplot') 
plt.show();

lasso_reg = Lasso(alpha=1)
lasso_reg.fit(X_train , Y_train)
cross_validation_accuracy = cross_val_score(estimator = lasso_reg, X = X_train, y = Y_train, cv = 5,verbose = 1)
y_pred = lasso_reg.predict(X_test)

print('\nLasso Regression')
print('Accuracy Score : %.6f' % lasso_reg.score(X_test, Y_test))

print('\nCross Validation Accuracy')
print(cross_validation_accuracy)

rmse = mean_squared_error(Y_test, y_pred)**0.5
r2 = r2_score(Y_test, y_pred)

print('\nMetrics')
print('RMSE   : %0.6f ' % rmse)
print('R2     : %0.6f ' % r2)

R2_Accuracy.append(r2);

# Putting together the coefficient and their corrsponding variable names  
lasso_reg_coefficient = pd.DataFrame() 
lasso_reg_coefficient["Columns"] = X_train.columns 
lasso_reg_coefficient['Coefficient Estimate'] = pd.Series(lasso_reg.coef_) 
print(lasso_reg_coefficient) 

# Let’s plot a bar chart of above coefficients using matplotlib plotting library.
# plotting the coefficient score 
fig, ax = plt.subplots(figsize =(10, 5)) 
  
color =['tab:gray', 'tab:blue', 'tab:orange',  
'tab:green', 'tab:red', 'tab:purple', 'tab:brown',  
'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',  
'tab:orange', 'tab:green', 'tab:blue', 'tab:olive'] 
  
ax.bar(lasso_reg_coefficient["Columns"],  
lasso_reg_coefficient['Coefficient Estimate'],color = color) 
  
ax.spines['bottom'].set_position('zero') 
  
plt.style.use('ggplot') 
plt.show(); 
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X_train , Y_train)
cross_validation_accuracy = cross_val_score(estimator = ridge_reg, X = X_train, y = Y_train, cv = 5,verbose = 1)
y_pred = ridge_reg.predict(X_test)

print('\nRidge Regression')
print('Accuracy Score : %.6f' % ridge_reg.score(X_test, Y_test))

print('\nCross Validation Accuracy')
print(cross_validation_accuracy)

rmse = mean_squared_error(Y_test, y_pred)**0.5
r2 = r2_score(Y_test, y_pred)

print('\nMetrics')
print('RMSE   : %0.6f ' % rmse)
print('R2     : %0.6f ' % r2)

R2_Accuracy.append(r2);

# Putting together the coefficient and their corrsponding variable names  
ridge_reg_coefficient = pd.DataFrame() 
ridge_reg_coefficient["Columns"] = X_train.columns 
ridge_reg_coefficient['Coefficient Estimate'] = pd.Series(ridge_reg.coef_) 
print(ridge_reg_coefficient) 

# Let’s plot a bar chart of above coefficients using matplotlib plotting library.
# plotting the coefficient score 
fig, ax = plt.subplots(figsize =(10, 5)) 
  
color =['tab:gray', 'tab:blue', 'tab:orange',  
'tab:green', 'tab:red', 'tab:purple', 'tab:brown',  
'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',  
'tab:orange', 'tab:green', 'tab:blue', 'tab:olive'] 
  
ax.bar(ridge_reg_coefficient["Columns"],  
ridge_reg_coefficient['Coefficient Estimate'],color = color) 
  
ax.spines['bottom'].set_position('zero') 
  
plt.style.use('ggplot') 
plt.show(); 
elasticnet_reg = ElasticNet(alpha=1)
elasticnet_reg.fit(X_train , Y_train)
cross_validation_accuracy = cross_val_score(estimator = elasticnet_reg, X = X_train, y = Y_train, cv = 5,verbose = 1)
y_pred = elasticnet_reg.predict(X_test)

print('\nElasticNet Regression')
print('Accuracy Score : %.6f' % elasticnet_reg.score(X_test, Y_test))

print('\nCross Validation Accuracy')
print(cross_validation_accuracy)

rmse = mean_squared_error(Y_test, y_pred)**0.5
r2 = r2_score(Y_test, y_pred)

print('\nMetrics')
print('RMSE   : %0.6f ' % rmse)
print('R2     : %0.6f ' % r2)

R2_Accuracy.append(r2);

# Putting together the coefficient and their corrsponding variable names  
elasticnet_reg_coefficient = pd.DataFrame() 
elasticnet_reg_coefficient["Columns"] = X_train.columns 
elasticnet_reg_coefficient['Coefficient Estimate'] = pd.Series(elasticnet_reg.coef_) 
print(elasticnet_reg_coefficient) 

# Let’s plot a bar chart of above coefficients using matplotlib plotting library.
# plotting the coefficient score 
fig, ax = plt.subplots(figsize =(10, 5)) 
  
color =['tab:gray', 'tab:blue', 'tab:orange',  
'tab:green', 'tab:red', 'tab:purple', 'tab:brown',  
'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',  
'tab:orange', 'tab:green', 'tab:blue', 'tab:olive'] 
  
ax.bar(elasticnet_reg_coefficient["Columns"],  
elasticnet_reg_coefficient['Coefficient Estimate'],color = color) 
  
ax.spines['bottom'].set_position('zero') 
  
plt.style.use('ggplot') 
plt.show();
dt_reg = DecisionTreeRegressor(random_state = 0)
dt_reg.fit(X_train , Y_train)
cross_validation_accuracy = cross_val_score(estimator = dt_reg, X = X_train, y = Y_train, cv = 5,verbose = 1)
y_pred = dt_reg.predict(X_test)

print('\nDecision Tree Regression')
print('Accuracy Score : %.6f' % dt_reg.score(X_test, Y_test))

print('\nCross Validation Accuracy')
print(cross_validation_accuracy)

rmse = mean_squared_error(Y_test, y_pred)**0.5
r2 = r2_score(Y_test, y_pred)

print('\nMetrics')
print('RMSE   : %0.6f ' % rmse)
print('R2     : %0.6f ' % r2)

R2_Accuracy.append(r2);


rf_reg = RandomForestRegressor()
rf_reg.fit(X_train , Y_train)
cross_validation_accuracy = cross_val_score(estimator = rf_reg, X = X_train, y = Y_train, cv = 5,verbose = 1)
y_pred = rf_reg.predict(X_test)

print('\nRandom Forest Regression')
print('Accuracy Score : %.6f' % rf_reg.score(X_test, Y_test))

print('\nCross Validation Accuracy')
print(cross_validation_accuracy)

rmse = mean_squared_error(Y_test, y_pred)**0.5
r2 = r2_score(Y_test, y_pred)

print('\nMetrics')
print('RMSE   : %0.6f ' % rmse)
print('R2     : %0.6f ' % r2)

R2_Accuracy.append(r2);
knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train , Y_train)
cross_validation_accuracy = cross_val_score(estimator = knn_reg, X = X_train, y = Y_train, cv = 5,verbose = 1)
y_pred = knn_reg.predict(X_test)

print('\nKNeighbor Regression')
print('Accuracy Score : %.6f' % knn_reg.score(X_test, Y_test))

print('\nCross Validation Accuracy')
print(cross_validation_accuracy)

rmse = mean_squared_error(Y_test, y_pred)**0.5
r2 = r2_score(Y_test, y_pred)

print('\nMetrics')
print('RMSE   : %0.6f ' % rmse)
print('R2     : %0.6f ' % r2)

R2_Accuracy.append(r2);
ada_reg = AdaBoostRegressor(n_estimators=500)
ada_reg.fit(X_train , Y_train)
cross_validation_accuracy = cross_val_score(estimator = ada_reg, X = X_train, y = Y_train, cv = 5,verbose = 1)
y_pred = ada_reg.predict(X_test)

print('\nAdaBoost Regressor')
print('Accuracy Score : %.6f' % ada_reg.score(X_test, Y_test))

print('\nCross Validation Accuracy')
print(cross_validation_accuracy)

rmse = mean_squared_error(Y_test, y_pred)**0.5
r2 = r2_score(Y_test, y_pred)

print('\nMetrics')
print('RMSE   : %0.6f ' % rmse)
print('R2     : %0.6f ' % r2)

R2_Accuracy.append(r2);
compare = pd.DataFrame({'Algorithms' : models , 'R2_Accuracy' : R2_Accuracy})
compare.sort_values(by='R2_Accuracy' ,ascending=False)
sns.factorplot(x='Algorithms', y='R2_Accuracy' , data=compare, size=6 , aspect=4);
from IPython.display import Image
import os
#!ls ../input/
Image("../input/diamondimages/Output.PNG")
