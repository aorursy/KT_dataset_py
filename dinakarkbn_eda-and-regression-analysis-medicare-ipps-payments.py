# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import time  # monitoring time
import os # accessing directory structure 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt # plotting using matplotlib
import seaborn as sns # plotting using seaborn
%matplotlib inline
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler # scaling the data
from sklearn.model_selection import train_test_split # train test split
from sklearn.preprocessing import OrdinalEncoder # ordinal encoding of categorical columns
from sklearn.linear_model import LinearRegression # LLinear Regression
from sklearn.tree import DecisionTreeRegressor # Decision Treee Regression
from sklearn.svm import SVR # Support Vector Regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Convert dytpes of object columns to float
def convert_col_dtype(col):    
    col = col.apply(lambda x: x.strip('$'))
    col = col.apply(lambda x: x.replace(',', ""))
    col = col.apply(pd.to_numeric, errors='coerce')
    print(col.dtype)
    return col
#Identifying Outliers in Numeric columns using IQR (Inter Quartile Range) and Q1 (25% Quantile), Q3(75% Quantile).
def identify_outliers(col):    
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5*iqr
    upper_limit = q3 + 1.5*iqr
    return(col, q1, q3, iqr, lower_limit, upper_limit)
def fit_lr(X_train, y_train, X_test, y_test):
    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    model.fit(X_train, y_train)
    print('Model score on Train data:', model.score(X_train, y_train))
    print('Model score on Test data:', model.score(X_test, y_test))
    return model
def fit_dtr(X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=5, min_samples_split=2, min_samples_leaf=1, 
                                  random_state=7, max_features=None, max_leaf_nodes=None)
    model.fit(X_train, y_train)
    print('Model score on Train data:', model.score(X_train, y_train))
    print('Model score on Test data:', model.score(X_test, y_test))
    return model
def fit_svr(X_train, y_train, X_test, y_test):
    model = SVR(C=1.0, kernel='rbf', epsilon=0.1)
    model.fit(X_train, y_train)
    print('Model score on Train data:', model.score(X_train, y_train))
    print('Model score on Test data:', model.score(X_test, y_test))
    return model
def predict_values(model, X_test):
    y_pred = model.predict(X_test)
    print("Predicted values:")
    print(y_pred[0:5])
    return y_pred
def performance_metrics(y_test, y_pred):
    # r2_score
    print("R2: ",r2_score(y_test, y_pred))
    # mse
    print("MSE: ",mean_squared_error(y_test, y_pred))
    # rmse
    print("RMSE: ",np.sqrt(mean_squared_error(y_test, y_pred)))
    # mean_absolute error
    print("MAE: ",mean_absolute_error(y_test, y_pred))
# Medicare_Provider_Charge_Inpatient_DRGALL_FY2016.csv has 197283 rows
df = pd.read_csv('/kaggle/input/medicare-provider-inpatient/Medicare_Provider_Charge_Inpatient_DRGALL_FY2016.csv', low_memory=False)
# rows and columns in dataframe
print(df.shape)
# Check top 5 rows of dataframe
df.head()
# Check last 5 rows of dataframe
df.tail()
#Check if there are any null values
df.isna().sum()
# Check duplicate rows
df.duplicated().sum()
# Replace spaces in columns to _
print(df.columns)
df.columns = df.columns.str.replace(' ','_')
print(df.columns)
# Check datatypes of columns
df.dtypes
# Column list after converting spaces to _
df.columns
# Convert dytpes of object columns to float
df['Total_Discharges'] = convert_col_dtype(df['Total_Discharges'])
df['Average_Covered_Charges'] = convert_col_dtype(df['Average_Covered_Charges'])
df['Average_Total_Payments'] = convert_col_dtype(df['Average_Total_Payments'])
df['Average_Medicare_Payments'] = convert_col_dtype(df['Average_Medicare_Payments'])
# Check datatypes of columns
df.dtypes
# Numeric columns
num_columns = ['Provider_Id', 'Provider_Zip_Code','Total_Discharges', 'Average_Covered_Charges', 'Average_Total_Payments','Average_Medicare_Payments']
# Categorical columns
cat_columns = ['DRG_Definition', 'Provider_Name', 'Provider_Street_Address', 'Provider_City', 'Provider_State','Hospital_Referral_Region_(HRR)_Description']
df.describe().T
# Plot 5 point summary
df.describe().drop('count',axis=0).plot(figsize=(10,5))
plt.show()
#Checking for Outliers and identifying them by calling identify_outliers() function.
#observations below Q1- 1.5*IQR, or those above Q3 + 1.5*IQR  are defined as outliers.

for col in num_columns :
    col, q1, q3, iqr, lower_limit, upper_limit = identify_outliers(col)
    print("\nColumn name : {}\n Q1 = {} \n Q3 = {}\n IQR = {}".format(col, q1, q3, iqr))
    print(" Lower limit = {}\n Upper limit = {}\n".format(lower_limit, upper_limit))
    outlier_count = len(df.loc[(df[col] < lower_limit) | (df[col] > upper_limit)])
    if outlier_count != 0 :
        print(outlier_count, "OUTLIERS ARE PRESENT in {} column.".format(col))
        print("Outlier datapoints in {} column are:".format(col))
        print(np.array(df.loc[(df[col] < lower_limit) | (df[col] > upper_limit)][col]))
    else:
        print("OUTLIERS ARE NOT PRESENT in {} column\n".format(col))
#Visualizing Outliers in dataset using boxplot

fig, ax = plt.subplots(2,3,figsize=(20, 10))
for col,subplot in zip(num_columns,ax.flatten()) :
    sns.boxplot(x=df[[col]], width=0.5, color='orange', ax=subplot)
    subplot.set_title('Boxplot for {}'.format(col))
    subplot.set_xlabel(col)    
plt.show()
plt.xticks(rotation = 30, fontsize=10)
plt.yticks(fontsize=10)
plt.plot(df[num_columns].var(), color='green', marker='s',linewidth=2, markersize=5)
plt.yscale('log')
plt.show()
# Distribution of columns using dist plots
fig, ax = plt.subplots(2,3,figsize=(20, 10))
for col,subplot in zip(num_columns,ax.flatten()) :
    ax =sns.distplot(df[col], ax=subplot, hist_kws={'color':'red','alpha':1}, kde_kws={'color':'black', 'lw':2})
# Exponential Distribution of columns using dist plots
fig, ax = plt.subplots(2,3,figsize=(20, 10))
for col,subplot in zip(num_columns,ax.flatten()) :
    ax =sns.distplot(np.log(df[col]), ax=subplot, hist_kws={'color':'g','alpha':1}, kde_kws={'color':'black', 'lw':2})
for col in cat_columns :
    print(col,':', len(df[col].unique()))
plt.figure(figsize=(35, 10))
sns.countplot(df['Provider_State'])
plt.show()
plt.figure(figsize=(35, 10))
sns.countplot(df['Provider_State'], order = df['Provider_State'].value_counts().index, palette=sns.color_palette("plasma"))
plt.show()
sns.pairplot(vars=np.log(df[num_columns]).columns,data=df, diag_kind='kde')
plt.show()
corr = df[num_columns].corr()
corr.style.background_gradient(cmap='YlGnBu')
sns.heatmap(corr, annot=True)
plt.show()
# fit transform columns in df
for col in cat_columns:
    col_enc = col+'_enc'
    print(col, col_enc, len(df[col].unique()))
    d={}
    for i in range(0, len(df[col].unique())):
        #print(col)
        #print(col, df[col].value_counts().index[i], df[col].value_counts()[i], i+1)
        d[df[col].value_counts().index[i]] = i+1
    print("col dict:", col,d)
    df.replace({col:d}, inplace=True)
print(df.columns)
X = df.drop('Average_Medicare_Payments', axis=1)
y = df['Average_Medicare_Payments']
print('Shape of Feture-set : ', X.shape)
print('Shape of Target-set : ', y.shape)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.30, random_state=7)
print("Training Set Shape:\nFeatures : {0}  Target : {1}\n".format(X_train.shape, y_train.shape))
print("Test Set Shape:\nFeatures : {0}  Target : {1}".format(X_test.shape, y_test.shape))
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scl = scaler.transform(X_train)
X_test_scl = scaler.transform(X_test)
# fit and score model
lr_model_medicare = fit_lr(X_train_scl, y_train, X_test_scl, y_test)
# predict values
y_pred_lr_med =  predict_values(lr_model_medicare, X_test_scl)
# Evaluate model performance
performance_metrics(y_test, y_pred_lr_med)
# Regression Plot for Avg Medicare Payments with LR model
sns.regplot(y_test, y_pred_lr_med)
plt.show()
# fit and score model
dtr_model_medicare = fit_dtr(X_train_scl, y_train, X_test_scl, y_test)
# predict values
y_pred_dtr_med =  predict_values(dtr_model_medicare, X_test_scl)
# Evaluate model performance
performance_metrics(y_test, y_pred_dtr_med)
# Regression Plot for Avg Medicare Payments with LR model
sns.regplot(y_test, y_pred_dtr_med)
plt.show()
# fit and score model
svr_model_medicare = fit_svr(X_train_scl, y_train, X_test_scl, y_test)
# predict values
y_pred_svr_med =  predict_values(svr_model_medicare, X_test_scl)
# Evaluate model performance
performance_metrics(y_test, y_pred_svr_med)
# Regression Plot for Avg Medicare Payments with LR model
sns.regplot(y_test, y_pred_svr_med)
plt.show()
X = df.drop('Average_Total_Payments', axis=1)
y = df['Average_Total_Payments']

print('Shape of Feture-set : ', X.shape)
print('Shape of Target-set : ', y.shape)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.30, random_state=7)

print("Training Set Shape:\nFeatures : {0}  Target : {1}\n".format(X_train.shape, y_train.shape))
print("Test Set Shape:\nFeatures : {0}  Target : {1}".format(X_test.shape, y_test.shape))
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scl = scaler.transform(X_train)
X_test_scl = scaler.transform(X_test)
# fit and score model
lr_model_total = fit_lr(X_train_scl, y_train, X_test_scl, y_test)

# predict values
y_pred_lr_tot =  predict_values(lr_model_total, X_test_scl)

# Evaluate model performance
performance_metrics(y_test, y_pred_lr_tot)

# Regression Plot for Avg Medicare Payments with LR model
sns.regplot(y_test, y_pred_lr_tot)
plt.show()
# fit and score model
dtr_model_total = fit_dtr(X_train_scl, y_train, X_test_scl, y_test)

# predict values
y_pred_dtr_tot =  predict_values(dtr_model_total, X_test_scl)

# Evaluate model performance
performance_metrics(y_test, y_pred_dtr_tot)

# Regression Plot for Avg Medicare Payments with LR model
sns.regplot(y_test, y_pred_dtr_tot)
plt.show()
