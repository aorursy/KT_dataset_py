# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/initial-data-exploration-in-python/"))
print(os.listdir("../input/house-prices-advanced-regression-techniques/"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/initial-data-exploration-in-python/transformed_pca.csv')
train.head()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = train.iloc[:,:-1]
y = train.SalePrice
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
error_df = pd.DataFrame({'actual':list(y_test),
                        'predicted':predictions})
error_df['error'] = error_df.actual - error_df.predicted
error_df.head()
from sklearn.metrics import r2_score,mean_squared_error

r_2 = r2_score(error_df.actual,error_df.predicted)
print('R-squared (coefficient of determination): '+str(r_2))
print('Root mean-squared error: ' + str(mean_squared_error(error_df.actual,error_df.predicted) ** (1/2)))
adj_r2= 1 - (1-r_2)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print('Adjusted R-squared: '+str(adj_r2))
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Plot the residuals after fitting a linear model
fig,ax = plt.subplots(figsize=(10,7),sharex=False)
sns.distplot(error_df.error, ax=ax)
ax.set_title('Error Distribution',fontdict=dict(size=18,weight='bold'))
ax.set_xlabel('')
plt.show();
# Use JointGrid directly to draw a custom plot
fig,ax = plt.subplots(figsize=(10,7))
p1 = sns.relplot(x="actual", y="error",data=error_df,ax=ax)
ax.set_title('Actual vs. Error',fontdict=dict(size=18,weight='bold'))
plt.close(p1.fig)
plt.show();
# optimization code from https://www.kaggle.com/dfitzgerald3/optimizing-ridge-regression-parameterization

from sklearn.linear_model import Ridge,RidgeCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score

scorer = make_scorer(mean_squared_error, False)

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
solvers = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']

cv_score = []
for i in solvers:
    for ii in alphas:
        clf = Ridge(alpha = ii, solver = i)
        cv_score.append([i,ii,np.sqrt(-cross_val_score(estimator=clf, 
                                            X=X_train, 
                                            y=y_train, 
                                            cv=15, 
                                            scoring = "neg_mean_squared_error")).mean()])

ridge_opt_df = pd.DataFrame(cv_score,columns=['solver','alpha','cv_score'])
ridge_opt_df.loc[ridge_opt_df.cv_score == ridge_opt_df.cv_score.min()]
r_reg = RidgeCV(alphas=[5],gcv_mode=['svd'],cv=15)
r_reg.fit(X_train,y_train)
predictions = r_reg.predict(X_test)
error_df = pd.DataFrame({'actual':list(y_test),
                        'predicted':predictions})
error_df['error'] = error_df.actual - error_df.predicted
error_df.head()
r_2 = r2_score(error_df.actual,error_df.predicted)
print('R-squared (coefficient of determination): '+str(r_2))
print('Root mean-squared error: ' + str(mean_squared_error(error_df.actual,error_df.predicted) ** (1/2)))
adj_r2= 1 - (1-r_2)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print('Adjusted R-squared: '+str(adj_r2))
fig,ax = plt.subplots(figsize=(10,7),sharex=False)
sns.distplot(error_df.error, ax=ax)
ax.set_title('Error Distribution',fontdict=dict(size=18,weight='bold'))
ax.set_xlabel('')
plt.show();
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train = train[[c for c in train.columns if c != 'SalePrice']]
test = pd.concat([train,test])
test_cleaned = pd.get_dummies(test[[c for c in test if c != 'MiscVal']],dummy_na=False,drop_first=True)
test_cleaned_cols = test_cleaned.columns
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
new_data = pd.DataFrame(my_imputer.fit_transform(test_cleaned))
new_data.columns = test_cleaned.columns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

scaler = MinMaxScaler(feature_range=[0, 1])
new_data = scaler.fit_transform(new_data[[c for c in new_data.columns if c != 'SalePrice']])
#Fitting the PCA algorithm with our Data
new_data = pd.DataFrame(new_data,columns=[c for c in test_cleaned_cols if c != 'SalePrice'])
pca = PCA().fit(new_data)
explained_var = pd.DataFrame({'n_components':list(np.arange(new_data.shape[1])),
                              'cumsum':list(np.cumsum(pca.explained_variance_ratio_))})
explained_var.loc[explained_var['cumsum'] > .99].iloc[:5,:]
pca = PCA(n_components=158)
transformed_features = pca.fit_transform(new_data)
transformed_data = pd.DataFrame(transformed_features)
transformed_data.shape
r_reg = RidgeCV(alphas=[5],gcv_mode=['svd'],cv=15)
r_reg.fit(transformed_data.iloc[:1460,:] ,y)
predictions = r_reg.predict(transformed_data.iloc[:1460,:])
error_df = pd.DataFrame({'actual':list(y),
                        'predicted':predictions})
error_df['error'] = error_df.actual - error_df.predicted
error_df.head()
r_2 = r2_score(error_df.actual,error_df.predicted)
print('R-squared (coefficient of determination): '+str(r_2))
print('Root mean-squared error: ' + str(mean_squared_error(error_df.actual,error_df.predicted) ** (1/2)))
adj_r2= 1 - (1-r_2)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print('Adjusted R-squared: '+str(adj_r2))
fig,ax = plt.subplots(figsize=(10,7),sharex=False)
sns.distplot(error_df.error, ax=ax)
ax.set_title('Error Distribution',fontdict=dict(size=18,weight='bold'))
ax.set_xlabel('')
plt.show();
predictions = r_reg.predict(transformed_data.iloc[1460:,:])
predictions = [np.exp(pred) for pred in predictions]
len(predictions)
pred_df = pd.DataFrame({'Id':np.arange(1461,transformed_data.shape[0]+1),
                       'SalePrice':predictions})
pred_df.to_csv('predictions.csv',index=False)