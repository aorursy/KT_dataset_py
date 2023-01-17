%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import skew
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import sklearn.preprocessing as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LassoCV

df_train = pd.read_csv('../input/train.csv', index_col="Id")
df_test = pd.read_csv('../input/test.csv', index_col="Id")
# Integrate train data and test data 
df_all = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'], df_test.loc[:,'MSSubClass':'SaleCondition']))
df_train.head()
plt.scatter(df_train.GrLivArea, df_train.SalePrice)
# Remove it as an outlier if the area is 4000 or more and the price is 200000 or less
df_train = df_train[~((df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 300000))]
# Update df_all
df_all = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'], df_test.loc[:,'MSSubClass':'SaleCondition']))
# Check the graph
plt.scatter(df_train.GrLivArea, df_train.SalePrice)
df_train.dtypes
nans = pd.concat([df_train.dtypes, df_train.isnull().sum(), df_test.isnull().sum()], axis=1, keys=['Type', 'Train', 'Test'])
# show list of missing values
nans[(nans.sum(axis = 1) > 0) & (nans['Type'] != "object") ]
# Filling missing values
df_all = df_all.fillna(df_all.mean())
# Visualization of correlation coefficient matrix
fig, ax = plt.subplots(1, 1, figsize=(30, 30))
sns.heatmap(df_train.corr(), vmax=1, vmin=-1, center=0, annot=True, ax=ax)
# If you look at this graph, You can pick up the variable of high multi-collinearity.
# But if there are a lot of variable, it is too difficult for you to find high multi-collinearty variable.
# Below this, it is a sample code of how to pick up variables.
indexs = df_train.corr().index
cols = []
row_count = 0
for row in df_train.corr().values:
    col_count = 0
    for col in row:
        # If the column and row are the same index, go to the next line.
        if row_count == col_count:
            break
        # SalePrice is not deleted due to purpose variable.
        if indexs[row_count] == "SalePrice":
            break
        # An index with an absolute value of multiple collinearity of 0.75 or more
        if abs(col) > 0.75:
            cols.append(indexs[row_count])
            break
        col_count+=1
    row_count+=1

print(cols)
df_all.drop(cols, axis=1, inplace=True)
# SalePrice's histogram
df_train["SalePrice"].hist(bins=30)
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
# Check SalePrice's histogram
df_train["SalePrice"].hist(bins=30)
# The fact that the histogram is downward means that the skewness is high.
# It is slightly rough, but if the skewness is 0.6 or more logarithmically transform.
non_categoricals = df_all.dtypes[df_all.dtypes != "object"].index
skewed_feats = df_train[non_categoricals].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.6].index
#Logarithmic transformation of features with skewness greater than 0.6
df_all[skewed_feats] = np.log1p(df_all[skewed_feats])
# Create a categorical variable list
categoricals = df_all.dtypes[df_all.dtypes == 'object'].index
categoricals_list = []
for index in categoricals:
    categoricals_list.append(index)
    
# Perform LabelEncoding and OneHotEncoding in a row
lbl = LabelEncoder()
ohe = OneHotEncoder(sparse = True)
nan_columns = []
for col in categoricals_list:
    lbl.fit(list(df_all[col].values.astype('str')))
    df_all[col] = lbl.transform(df_all[col].values.astype('str'))
    ohe.fit(df_all[col].values.reshape(1,-1).transpose())
    enced = ohe.transform(df_all[col].values.reshape(1,-1).transpose())
    column_names = []
    for cls in lbl.classes_:
        new_column_name = col + "_" + cls
        column_names.append(new_column_name)
        if cls == "nan":
            nan_columns.append(new_column_name)
    temp = pd.DataFrame(index=df_all.index, columns =column_names, data=enced.toarray())
    df_all = pd.concat([df_all, temp], axis=1)
    del df_all[col]
    
# Delete missing value column (_nan)
df_all.drop(nan_columns, axis=1, inplace=True)
# Split between train data and test data.
X = df_all[:df_train.shape[0]]
X_for_test = df_all[df_train.shape[0]:]
y = df_train.SalePrice
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1111)

# Find optimum hyperparameter using LassoCV
reg = LassoCV(cv=5, alphas=[0.1, 0.01, 0.001, 0.0005, 0.00055, 0.0006, 0.00065], max_iter=5000)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("Best alpha", reg.alpha_, "Best RMSE:", rmse)
# Learning with all data
reg.fit(X,y)
# Since log1p was done, we take expm1
pred = np.expm1(reg.predict(X_for_test))
solution = pd.DataFrame({"id":df_test.index, "SalePrice":pred})
solution.to_csv("test_lasso.csv", index = False)
coef = pd.Series(reg.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and deleted the other " +  str(sum(coef == 0)) + " variables.")