# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
# import pandas as pd
# import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
data = pd.read_csv("../input/train_V2.csv")
data.head()
data.columns
missing = {}
for column in data.columns:
    if data[column].isnull().sum()>0:
        missing['column'] = column
        missing['missing_values_count'] = data[column].isnull().sum()
        missing['percentage'] = data[column].isnull().sum()/len(data)*100
missing_df = pd.DataFrame(missing, index=[0])
missing_df
data[data['winPlacePerc'].isnull()]
data.dropna(inplace=True)
data.info()
# add features to quant list if they are int or float type
quant = [f for f in data.columns if data.dtypes[f] != 'object']
# add features to qualitative list if they are object type
qual = [f for f in data.columns if data.dtypes[f] == 'object']
quant
qual
for column in data.columns:
    print(column)
    print(len(data[column].unique()))
#     print(data[column].unique())
# qual.remove(['Id', 'groupId', 'matchId'])
qual = list(set(qual).difference(set(['Id', 'groupId', 'matchId'])))
target = data['winPlacePerc']
target_col = 'winPlacePerc'
quant.remove('winPlacePerc')
figsize(8,8)

plt.hist(target, bins = 100, edgecolor = 'k');
plt.xlabel('Win Percentage'); plt.ylabel('Count'); 
plt.title('Win Percentage Distribution');
def encode(frame, feature):
    ordering = pd.DataFrame()
    # extracting unique values from a feature(column)
    ordering['val'] = frame[feature].unique()
    # assigning the unique values to the index of the dataframe
    ordering.index = ordering.val
    # creating a column ordering with values assinged from 1 to the number of unique values
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    # creating a dict with the unique values as keys and the corresponding 
    # numbers in the ordering column as values
    ordering = ordering['ordering'].to_dict()
    # adding the encoded values into the original dataframe within new columns for each feature 
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o
    
qual_encoded = []
# encoding all the features in the qualitative list
for q in qual:  
    encode(data, q)
    qual_encoded.append(q+'_E')
qual_encoded
features = quant + qual_encoded 
train_data = data[features]#.drop(target_col, axis=1)
train_data.columns
def feat_correlation(frame, features, target_col):
    corr = pd.DataFrame()
    corr['feature'] = features
    corr['target'] = [frame[f].corr(frame[target_col], 'spearman') for f in features]
    corr = corr.sort_values('target')
    print(corr)
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=corr, y='feature', x='target', orient='h')
    return corr
    
corr = feat_correlation(data, features, 'winPlacePerc')
corr[corr.target > 0.2].feature
def corr_df(x, corr_val):
    '''
    Obj: Drops features that are strongly correlated to other features.
          This lowers model complexity, and aids in generalizing the model.
    Inputs:
          df: features df (x)
          corr_val: Columns are dropped relative to the corr_val input (e.g. 0.8)
    Output: df that only includes uncorrelated features
    '''

    # Creates Correlation Matrix and Instantiates
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterates through Correlation Matrix Table to find correlated columns
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index            
            val = item.values
            if val >= corr_val:
                # Prints the correlated feature set and the corr val
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(i)

    drops = sorted(set(drop_cols))[::-1]
    
    # Drops the correlated columns
    for i in drops:
        col = x.iloc[:, (i+1):(i+2)].columns.values
        df = x.drop(col, axis=1)

    return df
# Remove the collinear features above a specified correlation coefficient
features = corr_df(train_data, 0.6);
# modelxgb = XGBClassifier()
# modelxgb.fit(train_data[features], target)

# print(modelxgb.feature_importances_)
# from xgboost import plot_importance
# plot_importance(modelxgb)
# f_xgb = pd.DataFrame(data={'feature':features.columns,'value':modelxgb.feature_importances_})
# f_xgb = f_xgb.sort_values(['value'],ascending=False )
# plt.figure(figsize=(15,8))
# sns.barplot(f_xgb['feature'],f_xgb['value'])
# etcmodel = ExtraTreesClassifier()
# etcmodel.fit(features,target)
# print(etcmodel.feature_importances_)
# f_etc = pd.DataFrame(data={'feature':features.columns,'value':etcmodel.feature_importances_})
# f_etc = f_etc.sort_values(['value'],ascending=False )
# plt.figure(figsize=(15,8))
# sns.barplot(f_etc['feature'],f_etc['value'])
# ft = pd.merge(f_xgb, f_etc, how='inner', on=["feature"])
# ft.sort_values(["value_x","value_y"],ascending=False, inplace=True)
# top15ft = ft.head(15)
# top15ft
# ??
x_train, x_test, y_train, y_test = train_test_split(train_data[features],\
                                                    target, test_size = 0.2, random_state=42)   
# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))
baseline_guess = np.median(y_test)

print('The baseline guess is a score of %0.2f' % baseline_guess)
print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))
model_lr = LinearRegression(n_jobs=-1)
model_lr.fit(x_train, y_train)
y_lr = model_lr.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
mean_squared_error(y_test,y_lr)
r2_score(y_test, y_lr)
figsize(8,8)

plt.hist(y_lr, bins = 100, edgecolor = 'k');
plt.xlabel('Predicted Win Percentage'); plt.ylabel('Count'); 
plt.title('Predicted Win Percentage Distribution');
y_test.describe()
pd.Series(y_lr).describe()
print("Linear Regression Performance on the test set: MAE = %0.4f" % mae(y_test, y_lr))
test_data = pd.read_csv("../input/test_V2.csv")
test_data.head()
# add features to quant list if they are int or float type
test_quant = [f for f in test_data.columns if test_data.dtypes[f] != 'object']
# add features to qualitative list if they are object type
test_qual = [f for f in test_data.columns if test_data.dtypes[f] == 'object']
test_qual = list(set(test_qual).difference(set(['Id', 'groupId', 'matchId'])))
test_qual_encoded = []
# encoding all the features in the qualitative list
for q in test_qual:  
    encode(test_data, q)
    test_qual_encoded.append(q+'_E')
test_qual_encoded
test_features = test_quant + test_qual_encoded
test_features
# 
train_data.columns
model_lr_test = LinearRegression(n_jobs=-1)
model_lr_test.fit(train_data, target)
y_lr_pred = model_lr_test.predict(test_data[test_features])
submission = pd.DataFrame({'Id': test_data['Id'], 'winPlacePerc': list(y_lr_pred)})
y_lr
submission.to_csv("submission.csv",index=False)
