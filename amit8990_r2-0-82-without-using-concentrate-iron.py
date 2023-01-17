import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('/kaggle/input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv')
df
df.shape
df.info()
# Getting the column for the dataset
colname=df.columns
colname

# Correcting the integer formats
for col in colname:
    df[col]=df[col].str.replace(',','.')
df.head()
colname1=colname[1::]
colname1
# Converting string columns to numeric depending on the column types based on the technical details
for col in colname1:
    df[col]=df[col].apply(pd.to_numeric) 
df.info()
df['date'].value_counts()
df.describe()
# Outliers are checked for few numeric columns through Boxplots

plt.figure(figsize=(3600,2600),dpi=300)

df.plot(kind='box')
plt.show()
# Create a scatter plot to observe the distribution of silica % with time 
plt.figure(figsize=(20,5),dpi=100)
sns.scatterplot(x=df['date'],y=df['% Silica Concentrate'])
df=df.drop('date', axis=1)
df.corr()
# Creating function for outlier removal.
def remove_outlier(df_in, col_name): 
    Q1 = df_in[col_name].quantile(0.01) 
    Q3 = df_in[col_name].quantile(0.99) 
    IQR=Q3-Q1               # Interquantile range
    df_out = df_in.loc[(df_in[col_name] >= (Q1 - 1.5 * IQR)) & (df_in[col_name] <= (Q3 + 1.5 * IQR))]
    return(df_out)

# Removing outliers in numeric columns
for i in colname1:
    df= remove_outlier(df,i)
df.describe()
colname2=colname1[:-1]
colname2
# Checking the correlation between dependent and independent variables
for col in colname2:
    sns.pairplot(data=df,x_vars=col,y_vars=['% Silica Concentrate'])
plt.figure(figsize=(20, 20))

sns.heatmap(df[colname1].corr(), cmap="YlGnBu", annot = True)
plt.show()
# Dividing into X and Y sets for model building
# Putting target variable to y
y = df['% Silica Concentrate']

# Putting feature variables to X
X = df.drop(['% Silica Concentrate','% Iron Concentrate'], axis=1)


from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,test_size=0.3,random_state=100)
# instantiating an Standard Scaler object
scaler = StandardScaler()

# Scaling the numeric variables of train dataset
X_train[X_train.columns]= scaler.fit_transform(X_train[X_train.columns])
# Scaling the numeric variables of test dataset
X_test[X_test.columns]= scaler.transform(X_test[X_test.columns])
# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Running RFE with the output no variable of 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]
col
# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]
# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# checking VIF for the above model:
X_train_rfe = X_train_rfe.drop(['const'], axis=1)

# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_rfe.drop(["Flotation Column 03 Air Flow"], axis = 1)
X_train_new.columns
# Adding a constant variable 
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# checking VIF for the above model:

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
## Residual Analysis of the train data
y_train_pred= lm.predict(X_train_lm)
res=y_train - y_train_pred
# Importing the required libraries for plots.
import matplotlib.pyplot as plt
%matplotlib inline

# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((res), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
#Predicting for test values

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
# Making predictions
y_test_pred = lm.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_test_pred)
fig.suptitle('y_test vs y_pred_m', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label
from sklearn.metrics import r2_score
r2_score(y_true=y_test,y_pred=y_test_pred)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
# Instantiating PCA
pca = PCA(svd_solver='randomized', random_state=42)

# Performing PCA
pca.fit(X_train)
# Checking the principal components
pca.components_
# Checking the variance explained by principal components
pca.explained_variance_ratio_
#Scree plot for the explained variance
var_cumu = np.cumsum(pca.explained_variance_ratio_)
fig = plt.figure(figsize=[7,3])
plt.plot(range(1,len(var_cumu)+1), var_cumu)
plt.vlines(x=10, ymax=1, ymin=0, colors="r", linestyles="--")
plt.hlines(y=0.90, xmax=140, xmin=0, colors="g", linestyles="--")
plt.ylabel("Cumulative variance explained")
plt.xlabel("Number of components")
plt.show()
#### Applying the PCA on the train set

# Performing PCA with 28 components
pca_final = IncrementalPCA(n_components=10)
X_train_pca = pca_final.fit_transform(X_train)
X_train_pca.shape

# Applying the PCA transformation on the test set
X_test_pca = pca_final.transform(X_test)
X_test_pca.shape

# Adding a constant variable 
import statsmodels.api as sm  
X_train_pca= sm.add_constant(X_train_pca)
lm = sm.OLS(y_train,X_train_pca).fit()   # Running the linear model
#Let's see the summary of our linear model
print(lm.summary())
# Random forest
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestRegressor


cv_num =   5  #--> list of values

param={'max_depth': range(17,18,1)}


rf= RandomForestRegressor(warm_start=True)


model2 = GridSearchCV(estimator = rf, 
                        param_grid = param, 
                        scoring= 'r2', 
                        cv = cv_num, 
                        return_train_score=True,
                        verbose = 1)            
model2.fit(X_train, y_train) 

cv_results2 = pd.DataFrame(model2.cv_results_)
cv_results2 = cv_results2[cv_results2['param_max_depth']<=20]
cv_results2.head()
cv_results2['param_max_depth'] = cv_results2['param_max_depth'].astype('int32')

# plotting
plt.plot(cv_results2['param_max_depth'], cv_results2['mean_train_score'])
plt.plot(cv_results2['param_max_depth'], cv_results2['mean_test_score'])
plt.xlabel('max_depth')
plt.ylabel('Score')
plt.title("Score and param_max_depth")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
# printing the optimal accuracy score and hyperparameters
print('We can get auc of',model2.best_score_,'using',model2.best_params_)
# Importing XGboost classifier

import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance


# num_C = ______  #--> list of values
cv_num =   5 #--> list of values

param={'learning_rate': [0.01, 0.1, 0.3, 0.5], 
             'subsample': [0.3, 0.6, 0.9]}


# specify model
xgb1= xgb.XGBRegressor(max_depth=2, n_estimators=200)


model3 = GridSearchCV(estimator = xgb1, 
                        param_grid = param, 
                        scoring= 'r2', 
                        cv = cv_num, 
                        return_train_score=True,
                        verbose = 1)            
model3.fit(X_train, y_train) 
cv_results3 = pd.DataFrame(model3.cv_results_)

cv_results3.head()
# # plotting
plt.figure(figsize=(16,6))

param_grid = {'learning_rate': [0.01, 0.1, 0.3, 0.5], 
             'subsample': [0.3, 0.6, 0.9]} 

for n, subsample in enumerate(param['subsample']):
    
    # subplot 1/n
    plt.subplot(1,len(param['subsample']), n+1)
    df = cv_results3[cv_results3['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('r2')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.4, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')
# printing the optimal accuracy score and hyperparameters

print('We can get r2 of',model3.best_score_,'using',model3.best_params_)
final_model=RandomForestRegressor(max_depth=17,warm_start=True)
final_model.fit(X_train, y_train)
# Getting the predicted values on the test set and the r2 value
y_test_pred = final_model.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_true=y_test,y_pred=y_test_pred)
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize=20)         # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label