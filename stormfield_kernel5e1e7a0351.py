import pandas as pd

from pandas_profiling import ProfileReport

import missingno as msno

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

import seaborn as sns

import datetime as dt

import scipy.stats as ss
'''Plotly visualization .'''

import plotly.offline as py

#from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

#init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook
'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown
%matplotlib inline
plt.style.use('bmh')                    

sns.set_style({'axes.grid':False}) 
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
def bold(string):

    display(Markdown("**"+string+"**"))
base_dir_path = "/Users/zion/Documents/kaggle/house-prices-advanced-regression-techniques/"
training_data_path= base_dir_path + "train.csv"

test_data_path = base_dir_path + "test.csv"
profiling_output_path = base_dir_path + "tarining_data_profile.html"
training_data = pd.read_csv(training_data_path)

test_data  = pd.read_csv(test_data_path)

training_data.head()
test_data.head()
profile = ProfileReport(training_data)
profile.to_file(profiling_output_path)
msno.bar(training_data)
training_data.isnull().sum()
training_data.shape
test_data.isnull().sum()
features_to_drop = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
training_data = training_data.drop(features_to_drop, axis=1)
test_data = test_data.drop(features_to_drop, axis=1)
len_train_data = training_data.shape[0]

len_test_data =   test_data.shape[0]
training_data['SalePrice'].describe()
merged = pd.concat([training_data, test_data], axis = 0, sort = True)
merged.shape
merged.isnull().sum()  # to confirm
temp_num_features = training_data.select_dtypes(include = ['int64', 'float64']).copy()

correlaton_matrix = temp_num_features.corr()

#0.5 is the commmon base line for statistcs 

top_correlation_features = correlaton_matrix.index[abs(correlaton_matrix["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(temp_num_features[top_correlation_features].corr(),annot=True,cmap="RdYlGn")
sns.scatterplot(x="GrLivArea", y="SalePrice", data=training_data);
training_data[training_data['GrLivArea'] > 4000 ]
training_data.drop(training_data[training_data.GrLivArea > 4000 ].index, inplace=True)
merged_numerical_feature = merged.select_dtypes(include = ['int64', 'float64'])
merged_numerical_feature.shape #ie 38 out of 81
merged_numerical_feature.head()
merged_numerical_feature['MSSubClass'].value_counts()
merged_categorical_feature = merged.select_dtypes(include = ['object'])

merged_categorical_feature.head()
#check previous section

merged['MSSubClass']=merged['MSSubClass'].astype(str)
fig, axes = plt.subplots(nrows = 19, ncols = 2, figsize = (40, 200))

for ax, column in zip(axes.flatten(), merged_numerical_feature.columns):

    sns.distplot(merged_numerical_feature[column].dropna(), ax = ax, color = "steelblue")

    ax.set_title(column, fontsize = 43)

    ax.tick_params(axis = 'both', which = 'major', labelsize = 35)

    ax.tick_params(axis = 'both', which = 'minor', labelsize = 35)

    ax.set_xlabel('')

fig.tight_layout(rect = [0, 0.03, 1, 0.95])
merged.isnull().sum()[merged.isnull().sum()>0]
# for these use fill with- NONE/NA since it mean something - refer the data description

cols_to_impute_with_na = ['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual','FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType' ,'MasVnrType']

#for these use  median since they are numerical variables / descrete / countinous 

cols_to_impute_with_median = ['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','GarageArea', 'GarageCars', 'GarageYrBlt', 'MasVnrArea', 'TotalBsmtSF']

# for these use mode of the distribution - ie most common values - mostly catogorical

cols_to_impute_by_mode = ['Electrical','Exterior1st','Exterior2nd','Functional', 'KitchenQual', 'MSZoning','SaleType','Utilities']

merged['Electrical'].mode()
merged['Electrical'].mode()[0]
merged['BsmtFinSF1'].median()
for c in cols_to_impute_by_mode:

    merged[c].fillna(merged[c].mode()[0],inplace=True)
for c in cols_to_impute_with_na:

    merged[c].fillna('none',inplace=True)
for c in cols_to_impute_with_median:

    merged[c].fillna(merged[c].median(),inplace=True)
def cramers_v(confusion_matrix):

    """ calculate Cramers V statistic for categorial-categorial association.

        uses correction from Bergsma and Wicher,

        Journal of the Korean Statistical Society 42 (2013): 323-328

    """

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum()

    phi2 = chi2 / n

    r, k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
for col in merged_categorical_feature.columns:

    confusion_matrix = pd.crosstab(merged[col], merged["LotFrontage"]).as_matrix()

    value = cramers_v(confusion_matrix)

    print(f"{col}     {value}")
merged.groupby(['BldgType'])['LotFrontage'].describe() 
merged['LotFrontage'] = merged.groupby(['BldgType'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
merged.groupby(['BldgType'])['LotFrontage'].describe() # damn there is no median function.. hmm
merged.isnull().sum()[merged.isnull().sum()>0] #so all good! SalePrice-> missing is from the Test set
full_data = pd.get_dummies(merged)
full_data.shape
len_test_data , len_train_data
final_train=full_data[:len_train_data].copy()

final_test=full_data[len_test_data:].copy()
final_train.drop('Id', axis=1, inplace=True)

final_test.drop('Id', axis=1, inplace=True)
x=final_train.drop('SalePrice', axis=1)

y=final_train['SalePrice']

test_really_final=final_test.drop('SalePrice', axis=1)
from sklearn.preprocessing import  RobustScaler

sc=RobustScaler()

x=sc.fit_transform(x)

test=sc.transform(test_really_final)
l_model=Lasso(alpha =0.001, random_state=1)
l_model.fit(x,y)
pred=model.predict(test)

preds=np.exp(pred)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

mse = cross_val_score(lin_reg,x,y,scoring='neg_mean_squared_error',cv=5)

mse_mean = np.mean(mse)

print (mse_mean)

print(mse)
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

lasso = Lasso()

params = {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20,0.6]}

lasso_reg = GridSearchCV(lasso,params,scoring='neg_mean_squared_error',cv=5)

lasso_reg.fit(x,y)

print(lasso_reg.best_params_)

print(lasso_reg.best_score_)
print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
help(GridSearchCV)
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

ridge = Ridge()

params = {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20,0.6]}

ridge_reg = GridSearchCV(ridge,params,scoring='neg_mean_squared_error',cv=5)

ridge_reg.fit(x,y)

print(ridge_reg.best_params_)

print(ridge_reg.best_score_)
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from math import sqrt
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=40)

print(X_train.shape); print(X_test.shape)
lr = LinearRegression()

lr.fit(X_train, y_train)
pred_train_lr= lr.predict(X_train)

print(np.sqrt(mean_squared_error(y_train,pred_train_lr)))

print(r2_score(y_train, pred_train_lr))



pred_test_lr= lr.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,pred_test_lr))) 

print(r2_score(y_test, pred_test_lr))
rr = Ridge(alpha=0.01)

rr.fit(X_train, y_train) 

pred_train_rr= rr.predict(X_train)

print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))

print(r2_score(y_train, pred_train_rr))



pred_test_rr= rr.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,pred_test_rr))) 

print(r2_score(y_test, pred_test_rr))
model_lasso = Lasso(alpha=0.01)

model_lasso.fit(X_train, y_train) 

pred_train_lasso= model_lasso.predict(X_train)

print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))

print(r2_score(y_train, pred_train_lasso))



pred_test_lasso= model_lasso.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 

print(r2_score(y_test, pred_test_lasso))
pred=model_lasso.predict(test)

preds=np.exp(pred)
test_data.shape
preds.shape
test_data[1456:].shape
test_data[:1456].shape
test_data.shape
pred
#this was lasso

output=pd.DataFrame({'Id':test_data[:1456].Id, 'SalePrice':pred})

output.to_csv('submission.csv', index=False)