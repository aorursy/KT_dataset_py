#loading packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#graphs - boxplot
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid') 
import seaborn as sns
%matplotlib inline

#visulization
import plotly
from plotly.graph_objs import graph_objs as go
from IPython.html.widgets import interact

#Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
from pylab import rcParams

#ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Split datasets
from sklearn.model_selection import train_test_split

#Machine learning and error analysis
import xgboost as xgb
from sklearn.metrics import mean_squared_error


#Parameter tunning
from sklearn.model_selection import GridSearchCV

#Display an image
from IPython.display import Image

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#loading the datasets
dt_st = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv', sep=',')
dt_feat = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip', sep=',')
dt_train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip', sep=',')
dt_test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip', sep=',')
#Dataframe basic information

dt_st.info()
dt_feat.info()
dt_train.info()
dt_test.info()
dt_st.head(3)
dt_st.tail(3)
#Adjusting the format of the float
pd.set_option('display.float_format', lambda x: '%.2f' % x)
dt_st.describe()
dt_st.groupby('Type')['Size'].describe()
#to figure out how many Not avaliable (NA) variables we have in each column
for key,value in dt_st.iteritems():
    print(key,value.isnull().sum().sum())
bplot = sns.boxplot(y='Size', x='Type', data= dt_st, width=0.5, palette="bright")
bplot.axes.set_title("Boxplot of Stores: Type and distribution",fontsize=16)

#add swarplot
bplot=sns.swarmplot(y='Size', x='Type',data=dt_st, color='black', alpha=0.75)

#setting the axis
bplot.set_xlabel("Type",fontsize=12)
bplot.set_ylabel("Size",fontsize=12)
bplot.tick_params(labelsize=10)
dt_feat.head(3)
dt_feat.tail(3)
dt_feat.describe()
dt_feat.groupby('IsHoliday').describe(include=['object'])
#to figure out how many Not avaliable (NA) variables we have in each column
for key,value in dt_feat.iteritems():
    print(key,value.isnull().sum().sum())
plt.figure(figsize=(15, 10))
corr = dt_feat.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
#I am going to do a copy, because I don't want to do the features process now. I am just testing
dt_feat_copy = dt_feat.copy()
#I changed to datetime format, because I want to do a graph using 'date' as x-axis in datetime format
dt_feat_copy['Date'] = pd.to_datetime(dt_feat_copy['Date'])
#to figure out how many Not avaliable (NA) variables we have in each column
for key,value in dt_feat_copy.iteritems():
    print(key,value.isnull().sum().sum())
def f(var):
    plt.figure(figsize=(20,5))
    sns.lineplot(x="Date", y="{}".format(var), data=dt_feat_copy)
    
#if you are running this notebook on-line, run this function to make the things more intereative
#interact(f, var=dt_feat_copy[dt_feat_copy.columns[2:]]) 
list_1 = dt_feat_copy.columns[2:].tolist()
#version off-line - no interaction
data_1=[]
for i in list_1:
    data_1.append(f(i))
dt_train.head(3)
dt_train.tail(3)
for key,value in dt_train.iteritems():
    print(key,value.isnull().sum().sum())
dt_explor = dt_train.copy()
dt_explor = dt_explor.merge(dt_st, how='left').merge(dt_feat, how='left')
dt_explor['Date'] = pd.to_datetime(dt_explor['Date'])
dt_explor.head(3)
dt_explor.tail(3)
plt.figure(figsize=(20,5))
sns.lineplot(x="Date", y="Weekly_Sales", data=dt_explor)
plt.title('Weekly Sales')
#Let's find the total aggregate sales per date (observed) and transforms into a 'timeseries'
serie_1 = dt_explor.groupby(dt_explor['Date']).sum()['Weekly_Sales']
rcParams['figure.figsize'] = 11, 9
result = seasonal_decompose(serie_1, model='additive', freq=52) #weekly freq.
result.plot()
pyplot.show()
#selecting the period of the peaks.
peak_1 = (dt_explor['Date'] > '2010-11') & (dt_explor['Date'] <= '2010-12-26')
peak_2 = (dt_explor['Date'] > '2011-11') & (dt_explor['Date'] <= '2011-12-26')
#selecting the year of the peaks.
y_1 = (dt_explor['Date'] > '2010-1') & (dt_explor['Date'] < '2011-1')
y_2 = (dt_explor['Date'] > '2011-1') & (dt_explor['Date'] < '2012-1')
#sales in the peak 1 and its respective year (full) 
sales_1 = dt_explor.loc[peak_1, 'Weekly_Sales'].sum()
sales_total_1 = dt_explor.loc[y_1, 'Weekly_Sales'].sum()
#sales in the peak 2 and its respective year (full) 
sales_2 = dt_explor.loc[peak_2, 'Weekly_Sales'].sum()
sales_total_2 = dt_explor.loc[y_2, 'Weekly_Sales'].sum()
#Calculating the share
print('Share of peak 1: {}'.format(sales_1/sales_total_1)) 
print('Share of peak 2: {}'.format(sales_2/sales_total_2))
y_2011_without_christ = (dt_explor['Date'] > '2011-1') & (dt_explor['Date'] < '2011-11')
sales_withou_christ = dt_explor.loc[y_2011_without_christ, 'Weekly_Sales'].sum()
print('Sales w/t Christ: {}'.format(sales_withou_christ))
print('Sales in peak (Thanks Giving and Christ): {}'.format(sales_2))
print('Peak against rest of the year: {}'.format(sales_2/sales_withou_christ))
plot_acf(serie_1,lags=20)
plt.show()

plot_pacf(serie_1,lags=20)
plt.show()
#This function is going to be very useful to us, when we are looking for the total sum of sales by certain column 'd'.
def sales(s,d): return dt_explor.loc[dt_explor[d] == s, 'Weekly_Sales'].sum()
# Y is the column of the dataframe and X is the 'item' of this column that you want to find the share
def share_all(x,y): return sales(x,y)/dt_explor['Weekly_Sales'].sum()
dt_explor.groupby('Type')['Weekly_Sales'].sum()
#Share of sales by Store
print('Share A: {}'.format(share_all('A','Type')))
print('Share B: {}'.format(share_all('B','Type')))
print('Share C: {}'.format(share_all('C','Type')))
#function to find the total of sales by Type per date.
def type(x): 
    a = dt_explor.loc[dt_explor['Type'] == x]
    a = type_a = a.groupby(a['Date']).sum()['Weekly_Sales']
    return a
type_a = type('A')
rcParams['figure.figsize'] = 11, 9
result2 = seasonal_decompose(type_a, model='additive', freq=52)
result2.plot()
pyplot.show()
type_b = type('B')
type_c = type('C')
rcParams['figure.figsize'] = 11, 9
result2 = seasonal_decompose(type_b, model='additive', freq=52)
result2.plot()
pyplot.show()
rcParams['figure.figsize'] = 11, 9
result2 = seasonal_decompose(type_c, model='additive', freq=52)
result2.plot()
pyplot.show()
a = dt_explor.groupby(['Dept'])['Weekly_Sales'].sum()
plt.rcdefaults()
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize=(10,20))
ax.barh(a.index,a)
ax.set_yticks(a.index)
ax.set_yticklabels(a.index)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_ylabel('Dept')
ax.set_xlabel('Weekly Sales')
ax.set_title('Sum of Weekly Sales by Department', color='C0')

plt.show()
#How much the top 3 dept in sales (92, 95 and 38) represent in total sales? 
share_all(92,'Dept')+share_all(95,'Dept')+share_all(38,'Dept')

#lets take the dept 92 as example
dt_explor.loc[dt_explor['Dept']==92].head(3)
dt_explor.loc[dt_explor['Dept']==92].tail(3)
#function to find the total of observations of some dept by Type
def dept(i): 
    print(dt_explor.loc[dt_explor['Dept']==i].groupby(['Type'])['Dept'].count())
#Let's pick up some dept. randomly, just to check its frequency distribution
dept(92), dept(6), dept(23)
plt.figure(figsize=(15, 10))
corr = dt_explor.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap="YlGnBu",
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
#I will create a column to label our train and test dataset. It will make our lifes more easy at the time that we will split it
dt_train['label'] = 'train'
dt_test['label'] = 'test'
#Because I am going to concatenate, I will create a column of weekly sale in test set with n/a values, just to security
dt_test['Weekly_Sales'] = np.nan

dt_all = pd.concat([dt_train, dt_test])
dt_train.shape, dt_test.shape
dt_all.shape
dt_all.tail(3)
dt_all.head(3)
#let's merge
dt_all = dt_all.merge(dt_st, how='left').merge(dt_feat, how='left')
dt_all.shape
dt_all.head(3)
dt_all.tail(3)
dt_all['Date'] = pd.to_datetime(dt_all['Date'])
#To have a visualization of the space/dimension of the missing values into our dataset 
sns.heatmap(dt_all.notnull(), cbar=False, yticklabels='',cmap="Blues")
#Making a copy to preserve our original dataset
dt_all_interp = dt_all.copy()
#organising markdows using the average by store and temperature.
dt_all_interp_v2 = dt_all_interp.groupby(['Store','Dept','Temperature','IsHoliday']).median()[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].reset_index()
#then applying backward filling method 
dt_all_interp_v2 = dt_all_interp_v2.fillna(method='bfill')
#Checking null values
for key,value in dt_all_interp_v2.iteritems():
    print(key,value.isnull().sum().sum())
#Now I am going to interpolate CPI and Unemployment
dt_all_interp_v3 = dt_all_interp.groupby(['Store','Temperature']).median()[['CPI','Unemployment']].reset_index()
dt_all_interp_v3['CPI'] = dt_all_interp_v3['CPI'].interpolate()
dt_all_interp_v3['Unemployment'] = dt_all_interp_v3['Unemployment'].interpolate()
#Checking null values
for key,value in dt_all_interp_v3.iteritems():
    print(key,value.isnull().sum().sum())
#it's time to put it back into the full dataset. Before, let's rename the columns, because we do not want to replace all original data 
dt_all_interp_v2.rename(columns={'MarkDown1':'1_mk','MarkDown2':'2_mk','MarkDown3':'3_mk','MarkDown4':'4_mk','MarkDown5':'5_mk'}, inplace=True)
dt_all_interp_v3.rename(columns={'CPI':'inter_CPI','Unemployment':'inter_unempl'}, inplace=True)
#merging
dt_all =dt_all.merge(dt_all_interp_v2, on=['Store','Dept','Temperature','IsHoliday'], how = 'inner').merge(dt_all_interp_v3, on=['Store','Temperature'], how = 'inner')
#replacing
dt_all.MarkDown1.fillna(dt_all['1_mk'],inplace=True)
dt_all.MarkDown2.fillna(dt_all['2_mk'],inplace=True)
dt_all.MarkDown3.fillna(dt_all['3_mk'],inplace=True)
dt_all.MarkDown4.fillna(dt_all['4_mk'],inplace=True)
dt_all.MarkDown5.fillna(dt_all['5_mk'],inplace=True)
dt_all.CPI.fillna(dt_all['inter_CPI'],inplace=True)
dt_all.Unemployment.fillna(dt_all['inter_unempl'],inplace=True)
dt_all.drop(['1_mk','2_mk','3_mk','4_mk','5_mk','inter_CPI','inter_unempl'], axis=1, inplace=True)
#checking missing values
for key,value in dt_all.iteritems():
    print(key,value.isnull().sum().sum())
dt_all = pd.get_dummies(dt_all, columns=["Type",'IsHoliday'])
def dummy_92(c):
    if c['Dept'] == 92:
        return 1
    else:
        return 0

def dummy_6(c):
    if c['Dept'] == 6:
        return 1
    else:
        return 0

def dummy_23(c):
    if c['Dept'] == 23:
        return 1
    else:
        return 0
#toop 3 dept in sales
dt_all['dept_92'] = dt_all.apply(dummy_92, axis=1)
dt_all['dept_6'] = dt_all.apply(dummy_6, axis=1)
dt_all['dept_23'] = dt_all.apply(dummy_23, axis=1)
dt_all.head()
#excuding unsuless dummies
dt_all = dt_all.drop(columns=['Type_C','IsHoliday_False'])
for key,value in dt_all.iteritems():
    print(key,value.isnull().sum().sum())
dt_all.shape
#excluding variables
dt_all = dt_all.drop(columns=['Dept','Store','Size','MarkDown4'])
for key,value in dt_all.iteritems():
    print(key,value.isnull().sum().sum())
#spliting data
train = dt_all[dt_all.label=='train'].reset_index(drop=True)
test = dt_all[dt_all.label=='test'].reset_index(drop=True)
train = train.drop(columns=['label'])
test = test.drop(columns=['label'])
train.head()
train.shape, test.shape
#X_test
X_test = test.iloc[:,2:]
X_test.head(3)
#train
y_train,X_train =  train.iloc[:,1],train.iloc[:,2:]
#train_2 and validation set
X_train_2, X_valid, y_train_2, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=123)
Image('../input/models-sklearn/Screenshot from 2020-07-12 02-20-33.png')
# Create optimized DMatrix to improve the quality of model
sales_dmatrix = xgb.DMatrix(data=X_train_2,label=y_train_2)

#  Parameter dictionary for each tree: params - just a generic example without tunning
params = {"objective":"reg:linear", "max_depth":4}


#Let's start with parameter tuning by seeing how the number of boosting rounds (number of trees you build) impacts the out-of-sample performance of your XGBoost mode

# Perform cross-validation with early stopping
cv_results = xgb.cv(dtrain=sales_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

print(cv_results)
# Creating dict. of range of parameters to grid
gbm_param_grid = {
    'learning_rate': [0.01,0.1,0.5],
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# the regressor
gbm = xgb.XGBRegressor()

# Grid search (yes!): 
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring='neg_mean_squared_error', cv=4, verbose=1)

#Fit the parameters!
grid_mse.fit(X_train_2, y_train_2)

# best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
# Now, back to the model these new parameters
xg_reg_2 = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.7, learning_rate = 0.5, max_depth = 5, alpha = 10, n_estimators = 50)
#fiting the model into our training dataset
xg_reg_2.fit(X_train_2,y_train_2)
#Feature importance
xgb.plot_importance(xg_reg_2)
plt.show()
#!pip install shap
#import shap
#shap.initjs()

# explain the model's predictions using SHAP

#explainer = shap.TreeExplainer(xg_reg_2)
#shap_values = explainer.shap_values(X_train_2,check_additivity=False)

# summarize the effects of all the features
#shap.summary_plot(shap_values, X_train_2)
Image('../input/shap-features/Screenshot from 2020-07-11 19-46-04.png')
#applying the model to make the predictions, based on the features of the validation dataset
preds = xg_reg_2.predict(X_valid)
#let's see the error - rmse 
rmse = np.sqrt(mean_squared_error(y_valid, preds))
print("RMSE: %f" % (rmse))
#function to measure the error, based on the criteria of this competion
def wmae(dataset, real, predicted):
    weights = dataset.IsHoliday_True.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)
wmae(X_valid,y_valid,preds)
#forecast in test set
preds_final = xg_reg_2.predict(X_test)
#Preparing the subimission
dt_submission = dt_test.copy()
dt_submission['weeklySales'] = preds_final
#adapting the model
dt_submission['id'] = dt_submission['Store'].astype(str) + '_' +  dt_submission['Dept'].astype(str) + '_' +  dt_submission['Date'].astype(str)
dt_submission = dt_submission[['id', 'weeklySales']]
dt_submission = dt_submission.rename(columns={'id': 'Id', 'weeklySales': 'Weekly_Sales'})
dt_submission.head(3)
dt_submission.info()
dt_submission.to_csv('output_submission.csv', index=False)