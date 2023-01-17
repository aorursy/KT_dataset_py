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
#import files that are required for reading the data. 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
#plt.figure(figsize=(16,5))

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
df =  pd.read_csv("../input/regression-with-neural-networking/concrete_data.csv")
df.head()
## Data Visualization and Data cleaning
df.info()
# reduce file size. 

col = list(df.columns)
col.remove('Age')


for n in col:
    df[n]= df[n].astype(np.float32)
    
df.info()
    
df.describe()
#what are the combinations needed to achieve max strength?
df[df['Strength'] >= 80]

# Only 3 rows are listed for max strength. May be target strength range (Say 65 to 83) to made wider inorder to get optimium combinations. 
g = sns.PairGrid(df)
g.map(plt.scatter)
g = sns.PairGrid(df)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
# variables Blast furance slag, fly ash & superplasticizer are all dense at zero value. The degree of influence of these variables
# varies significantly! 
df_non_zero = df[~(df['Fly Ash']== 0) & ~(df['Blast Furnace Slag'] == 0) & ~(df['Superplasticizer'] == 0)]
g = sns.PairGrid(df_non_zero)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g = sns.JointGrid(x="Water", y="Strength", data=df_non_zero)
g = g.plot(sns.regplot, sns.distplot)

## Data analysis
# import sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
def standardize( num_list):
        
    list_std= []
    nl_mean = np.mean(num_list, dtype = np.float64)
    nl_std  = np.std(num_list, dtype = np.float64)
    for i in num_list:
        list_std.append((i- nl_mean)/ nl_std)
    return (np.round(list_std, 2))
   
scale_col=list(df.columns)
scale_col.remove('Strength')

#standardize all the values of X and scale only y target values. 
for n in scale_col:
    df[n]= standardize(list(df[n].values))

df.head()
def y_data(y):
    '''
    input = y array
    
    output:
    y(i) = y(i) - ymean
    
    '''
    y_list = []
    y_mean = np.mean(y, dtype = np.float32)
    
    for i in y:
        y_list.append((i- y_mean))
    return (np.round(y_list, 4))
df['Strength']= y_data(df['Strength'].values)

df_scaled = df.copy()
# dataframe split into x and y data
X = df_scaled.drop(['Strength'], axis = 1)
y = df_scaled['Strength'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
method = SelectKBest(score_func= mutual_info_regression, k = 'all')

method.fit_transform(X_train, y_train)
correlation_matrix = X_train.corr(method= 'pearson').abs()
upper_corr_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
#print(upper_corr_matrix)
plt.figure(figsize=(16,5))
sns.heatmap(data = upper_corr_matrix , cmap= 'YlGnBu', annot= True)
# filter the columns which have greater than 0.5 correlation !

to_filter = [column for column in upper_corr_matrix.columns if any (upper_corr_matrix[column] > 0.20)]
to_filter
# new reduced x input. 
X_new = df_scaled[to_filter]
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, max_error,median_absolute_error, mean_squared_log_error

# function to evaulate performance of the regressor. 

def evaulation(model, y_pred, y_true):
    
    '''
    Input:- model = string (Name of the regressor)
    y_pred= model prediction
    y_true = actual labels. 
    
    Output:
    Dataframe with evaulation matrix. 
    
    '''
    
    # create data output frame for the evaluation. 
    data = [explained_variance_score(y_true,y_pred), 
            max_error(y_true,y_pred),
            mean_squared_error(y_true,y_pred),
            mean_absolute_error(y_true,y_pred),
            r2_score(y_true,y_pred, multioutput='uniform_average'),
            median_absolute_error(y_true,y_pred)           
            ]
    row_index = ['Exp_Var_Score', 'Max_Error','MSE','MAE','R2_Score', 'Median_Abs_Error']
    
    df = pd.DataFrame(data, columns= [model], index= row_index)
    
    return df
# Step1 Train test split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state = 0)

# Step2 Initiate linear regressor
lr = LinearRegression()

# step3 fit the data
lr.fit(X_train, y_train)

# predict the test data
y_pred_lr = lr.predict(X_test)


#evaulation of the lr   

print('Intercept:', lr.intercept_)
print('Coefficients:', lr.coef_)

# Linear regressor evaulation parameters
df_linear = evaulation('linear', y_pred_lr, y_test)
df_linear
# Linear regression has high MSE and low R2_score. Hence this regression is not a great model. 
## SVR
from sklearn.metrics import r2_score
# default parameters before running gridsearch. 
svr = SVR(C=100, epsilon=0.9, kernel='rbf', gamma= 3, tol = 1e-6)


pipe = Pipeline( steps = [('MinMax', MinMaxScaler()), 
                          ('SVR', svr)])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)



#evaulation of the Ridge  
svr_reg = np.mean((y_pred - y_test)**2)
svr_reg_max = np.max((y_pred - y_test)**2)
svr_reg_min = np.min((y_pred - y_test)**2)

print('MSE:', svr_reg)
print('MSE Max:', svr_reg_max)
print('MSE Min:', svr_reg_min)
print('R2_score:', r2_score(y_test, y_pred))

# intercept and coefficients are available only for kernel = 'linear'
# SVR has better MSE and R2 scores compared to linear regression model. 
# Linear regressor evaulation parameters
df_svr = evaulation('SVR', y_pred, y_test)
df_svr
# To see how the prediction data looks like.
sample = X_test[:10]
sample_y_test = y_test[:10]
y_pred_sample = pipe.predict(sample)
data = { 'Pred': y_pred_sample, 'Actual': sample_y_test.values}
sample_df = pd.DataFrame(data, columns = ['Pred', 'Actual'])
sample_df
# index[0] & [3] has high error!
## Conclusions
# SVR predictions are close to actual values. R2 score for the SVR is almost 0.83. This value should be looked along with MSE, 
# the error is still high. The data has high variance need to be careful while predicting the target value. 
