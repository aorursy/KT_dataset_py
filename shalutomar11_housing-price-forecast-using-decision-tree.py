import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
#from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
# reading from csv file
data_with_no_data_frame = pd.read_csv("../input/housing-dataset/HousingData.csv")
data = pd.DataFrame(data_with_no_data_frame)
# descriptive statistics
data.describe().transpose()
data.dtypes #types of data
# CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, PTRATIO,B,LSTAT,MEDV(float- continous data)  RAD, TAX(int- discrete data)
# reading top 5 eleemts
data.head(5)
# found nan coloumns in data sets
data.isnull().sum()
data_feature = data.columns
data_feature
data.hist(figsize=(8,8),bins=50)
# mean is best values to replace NaN values (median will return me highest values and mode most return values will be 0) 
mean_CRIM = data['CRIM'].mean(skipna=True)
print(mean_CRIM)
data['CRIM'] = data['CRIM'].mask(data['CRIM']==0).fillna(data['CRIM'].mean())
#data['CRIM'] = data.CRIM.mask(data.CRIM == 0,mean_CRIM)
print("number of NAN values: \t"+str(data['CRIM'].isnull().sum()))
plt.hist(data['CRIM'],color = 'blue', edgecolor = 'black')
plt.title('Distribution of CRIM')
plt.xlabel('CRIM')
plt.ylabel('count of CRIM')
mean_ZN = data['ZN'].mean(skipna=True)
print(mean_ZN)
data['ZN'] = data['ZN'].mask(data['ZN']==0).fillna(data['ZN'].mean())
print("number of NAN values: \t"+str(data['ZN'].isnull().sum()))
plt.hist(data['ZN'],color = 'blue', edgecolor = 'black')
plt.title('ZN Distribution plot')
plt.xlabel('ZN')
plt.ylabel('Count of ZN')
# mean is best values to replace NaN values (median will return me highest values and mode most return values will be 0) 
data['ZN'].median()
data['ZN'].mean()
data['ZN']=data['ZN'].fillna(data['ZN'].median(axis=0))
print("Number of NaN : \t"+str(data["ZN"].isnull().sum()))
sns.distplot(data['ZN'],fit=norm, kde=False)
data['ZN']
plt.hist(data['INDUS'],color = 'blue', edgecolor = 'black')
plt.title('INDUS Distribution plot')
plt.xlabel('INDUS')
plt.ylabel('Count of INDUS')
# mean is best values to replace NaN values (median will return me highest values and mode most return values will be 0) 
data['INDUS'].median()
data['INDUS'].mean()
data['INDUS'] = data['INDUS'].mask(data['INDUS']==0).fillna(data['INDUS'].mean())
print("Number of NaN : \t"+str(data["INDUS"].isnull().sum()))
sns.distplot(data['INDUS'],fit=norm, kde=False)
# mean is best values to replace NaN values (median will return me highest values and mode most return values will be 0) 
data['AGE'].median()
data['AGE'].mean()
data['AGE'] = data['AGE'].mask(data['AGE']==0).fillna(data['AGE'].median())
print("Number of NaN : \t"+str(data["AGE"].isnull().sum()))
sns.distplot(data['AGE'],fit=norm, kde=False)
# mean is best values to replace NaN values (median will return me highest values and mode most return values will be 0) 
data['LSTAT'].median()
data['LSTAT'].mean()
data['LSTAT'] = data['LSTAT'].mask(data['LSTAT']==0).fillna(data['LSTAT'].mean())
print("Number of NaN : \t"+str(data["LSTAT"].isnull().sum()))
sns.distplot(data['LSTAT'],fit=norm, kde=False)
# mean is best values to replace NaN values (median will return me highest values and mode most return values will be 0) 
data['CHAS'].median()
data['CHAS'].mean()
data['CHAS'] = data['CHAS'].mask(data['CHAS']==0).fillna(data['CHAS'].mean())
print("Number of NaN : \t"+str(data["CHAS"].isnull().sum()))
sns.distplot(data['CHAS'],fit=norm, kde=False)
# found nan coloumns in data sets
data.isnull().sum()
data['MEDV'] = np.where(data['MEDV']>31,data['MEDV'].mean(),data['MEDV'])
sns.boxplot(y='MEDV',data=data)
# reading from csv file
data_with_no_data_frame = pd.read_csv("../input/housing-dataset/HousingData.csv")
data1 = pd.DataFrame(data_with_no_data_frame)
# imputing technique can be done by package called Imputer , where data can be passed to fit_transform and replacing entire NaN with mean
imr = data.fillna(data.mean())
imputed_data = pd.DataFrame(imr,index=data.index,columns = data.columns)
imputed_data
# cmap - color of heatmap
# line width - 3 (black color lines)
# square - true # basically given heat will be rectangle box inside , it show the square
# line color - black
# cbar_kws - orienetation either horizontal or vertical(by default)
# annot = true -> To add text over the heatmap, we can use the annot attribute. If annot is set to True, the text will be written on each cell.
plt.figure(figsize=(10, 15))
sns.heatmap(data.corr().abs(),annot=True,vmin=-1, vmax=1, center= 0,cmap= 'coolwarm',linewidths=3, square=True,linecolor='black',cbar_kws= {'orientation': 'horizontal'})
# correlation with MEDV with rest of coloumn with 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
# 'PTRATIO', 'B', 'LSTAT']
data.corrwith(data['MEDV'], axis = 0)
# seaborn are used to visualize a linear relationship as determined through regression. These functions, regplot()
# LSTAT, INDUS, RM, TAX, NOX, PTRAIO
# best correlation variables regression plot 
fig, (ax1, ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(ncols=8, sharey=True)
column_sels = ['LSTAT', 'INDUS', 'NOX', 'CRIM', 'TAX','RAD', 'AGE']
sns.regplot(data['LSTAT'], data['MEDV'],ax=ax1)
sns.regplot(data['INDUS'], data['MEDV'],ax=ax2)
sns.regplot(data['NOX'], data['MEDV'],ax=ax3)
sns.regplot(data['CRIM'], data['MEDV'],ax=ax4)
sns.regplot(data['RM'], data['MEDV'],ax=ax5)
sns.regplot(data['TAX'], data['MEDV'],ax=ax6)
sns.regplot(data['RAD'], data['MEDV'],ax=ax7)
sns.regplot(data['AGE'], data['MEDV'],ax=ax8)
fig.set_size_inches(18.5, 10.5)
data.corrwith(data['MEDV'], axis = 0).plot(kind='barh')
# LSTAT, INDUS, RM, TAX, NOX, PTRAIO
fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(ncols=8, sharey=True)
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
sns.regplot(data['LSTAT'], data['MEDV'],ax=ax1)
sns.regplot(data['INDUS'], data['MEDV'],ax=ax2)
sns.regplot(data['NOX'], data['MEDV'],ax=ax3)
sns.regplot(data['PTRATIO'], data['MEDV'],ax=ax4)
sns.regplot(data['RM'], data['MEDV'],ax=ax5)
sns.regplot(data['TAX'], data['MEDV'],ax=ax6)
sns.regplot(data['DIS'], data['MEDV'],ax=ax7)
sns.regplot(data['AGE'], data['MEDV'],ax=ax8)
fig.set_size_inches(18.5, 10.5)

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in data.items():
    sns.boxplot(y=k,data=data, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
lm = LinearRegression()
# storing alcohol parameter in x and color intensity as y
x = data[['LSTAT', 'INDUS', 'NOX', 'CRIM', 'TAX','RAD', 'AGE']]
y = data['MEDV']
Y = y.values.reshape(-1,1)
# From given dataset the splitting the data into training and testing the data
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.3, random_state=0)
print(X_train.shape)
print(Y_train.shape)
lm.fit(X=X_train,y=Y_train)
# To retrieve the intercept:
print("intercept:  "+ str(lm.intercept_))

# For retrieving the slope:
print("slope: "+ str(lm.coef_))

# for x datapoint poviding y_pred value
y_pred = lm.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
# The mean squared error
print('Mean squared error: %.2f'% mean_squared_error(Y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination r2_score: %.2f'% r2_score(Y_test, y_pred))
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, train_test_split
dtm = DecisionTreeRegressor(max_depth=4,
                           min_samples_split=5,
                           max_leaf_nodes=10)
dtm.fit(x,y)
print("R-Squared on train dataset={}".format(dtm.score(X_test,Y_test)))

dtm.fit(X_test,Y_test)   
print("R-Squaredon test dataset={}".format(dtm.score(X_test,Y_test)))
print(x.shape,y.shape)
param_grid = {"criterion": ["mse", "mae"],
              "min_samples_split": [10, 20, 40],
              "max_depth": [2, 6, 8],
              "min_samples_leaf": [20, 40, 100],
              "max_leaf_nodes": [5, 20, 100],
              }

## Comment in order to publish in kaggle.

grid_cv_dtm = GridSearchCV(dtm, param_grid, cv=5)

grid_cv_dtm.fit(x,y)
print("R-Squared::{}".format(grid_cv_dtm.best_score_))
print("Best Hyperparameters::\n{}".format(grid_cv_dtm.best_params_))
df = pd.DataFrame(data=grid_cv_dtm.cv_results_)
df.head()
fig,ax = plt.subplots()
sns.pointplot(data=df[['mean_test_score',
                           'param_max_leaf_nodes',
                           'param_max_depth']],
             y='mean_test_score',x='param_max_depth',
             hue='param_max_leaf_nodes',ax=ax)
ax.set(title="Effect of Depth and Leaf Nodes on Model Performance")
# Checking the training model scores
r2_scores = cross_val_score(grid_cv_dtm.best_estimator_, x, y, cv=10)
mse_scores = cross_val_score(grid_cv_dtm.best_estimator_, x, y, cv=10,scoring='neg_mean_squared_error')

print("avg R-squared::{:.3f}".format(np.mean(r2_scores)))
print("MSE::{:.3f}".format(np.mean(mse_scores)))
best_dtm_model = grid_cv_dtm.best_estimator_

y_pred = best_dtm_model.predict(X_test)
residuals = Y_test.flatten() - y_pred


r2_score = best_dtm_model.score(X_test,Y_test)
print("R-squared:{:.3f}".format(r2_score))
print("MSE: %.2f" % metrics.mean_squared_error(Y_test, y_pred))