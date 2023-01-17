import time
start = time.time()
# # Installing Jupyter Notebook Extensions
# !pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
# !pip install jupyter_nbextensions_configurator
# !jupyter contrib nbextension install --user
# !jupyter nbextensions_configurator enable --user
%matplotlib inline
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize

# from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score

data = pd.read_csv("../input/train_V2.csv")#, nrows=100000)
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
# quant
# qual
# for column in data.columns:
#     print(column)
#     print(len(data[column].unique()))
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
# def feat_correlation(frame, features, target_col):
#     corr = pd.DataFrame()
#     corr['feature'] = features
#     corr['target'] = [frame[f].corr(frame[target_col], 'spearman') for f in features]
#     corr = corr.sort_values('target')
#     print(corr)
#     plt.figure(figsize=(6, 0.25*len(features)))
#     sns.barplot(data=corr, y='feature', x='target', orient='h')
#     return corr
    
# corr = feat_correlation(data, features, 'winPlacePerc')
# corr[corr.target > 0.2].feature
# def corr_df(x, corr_val):
#     '''
#     Obj: Drops features that are strongly correlated to other features.
#           This lowers model complexity, and aids in generalizing the model.
#     Inputs:
#           df: features df (x)
#           corr_val: Columns are dropped relative to the corr_val input (e.g. 0.8)
#     Output: df that only includes uncorrelated features
#     '''

#     # Creates Correlation Matrix and Instantiates
#     corr_matrix = x.corr()
#     iters = range(len(corr_matrix.columns) - 1)
#     drop_cols = []

#     # Iterates through Correlation Matrix Table to find correlated columns
#     for i in iters:
#         for j in range(i):
#             item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
#             col = item.columns
#             row = item.index            
#             val = item.values
#             if val >= corr_val:
#                 # Prints the correlated feature set and the corr val
#                 print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
#                 drop_cols.append(i)

#     drops = sorted(set(drop_cols))[::-1]
    
#     # Drops the correlated columns
#     for i in drops:
#         col = x.iloc[:, (i+1):(i+2)].columns.values
#         df = x.drop(col, axis=1)

#     return df
# # Remove the collinear features above a specified correlation coefficient
# corr = corr_df(train_data, 0.6);
# quant.remove('damageDealt')
# quant.remove('winPoints')
# features = quant + qual_encoded
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
# x_train, x_test, y_train, y_test = train_test_split(train_data[features],\
#                                                     target, test_size = 0.2, random_state=42)   
# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(x_train, y_train)
    
    # Make predictions and evalute
    model_pred = model.predict(x_test)
    model_mae = mae(y_test, model_pred)
    
    # Return the performance metric
    return model_mae
# baseline_guess = np.median(y_test)

# print('The baseline guess is a score of %0.2f' % baseline_guess)
# print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))
# %%time

# lr = LinearRegression(n_jobs=-1)
# lr_mae = fit_and_evaluate(lr)

# print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)

# figsize(8,8)

# plt.hist(y_lr, bins = 100, edgecolor = 'k');
# plt.xlabel('Predicted Win Percentage'); plt.ylabel('Count'); 
# plt.title('Predicted Win Percentage Distribution');
# y_test.describe()
# pd.Series(y_lr).describe()
# %%time

# svm = SVR(C = 1000, gamma = 0.1)
# svm_mae = fit_and_evaluate(svm)

# print('Support Vector Machine Regression Performance on the test set: MAE = %0.4f' % svm_mae)
# %%time 

# random_forest = RandomForestRegressor(random_state=42, n_jobs=-1)
# random_forest_mae = fit_and_evaluate(random_forest)

# print('Random Forest Regression Performance on the test set: MAE = %0.4f' % random_forest_mae)
# %%time

# gradient_boosted = GradientBoostingRegressor(random_state=42)
# gradient_boosted_mae = fit_and_evaluate(gradient_boosted)

# print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' % gradient_boosted_mae)
# %%time

# knn = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
# knn_mae = fit_and_evaluate(knn)

# print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f' % knn_mae)
# %%time

# plt.style.use('fivethirtyeight')
# figsize(8, 6)

# # Dataframe to hold the results
# model_comparison = pd.DataFrame({'model': ['Linear Regression', 'Random Forest',# 'Support Vector Machine',
#                                            'Gradient Boosted', 'K-Nearest Neighbors'],
#                                  'mae': [lr_mae, random_forest_mae,# svm_mae, 
#                                          gradient_boosted_mae, knn_mae]})

# # Horizontal bar chart of test mae
# model_comparison.sort_values('mae', ascending = False).plot(x = 'model', y = 'mae', kind = 'barh',
#                                                            color = 'red', edgecolor = 'black')

# # Plot formatting
# plt.ylabel(''); plt.yticks(size = 14); plt.xlabel('Mean Absolute Error'); plt.xticks(size = 14)
# plt.title('Model Comparison on Test MAE', size = 20);




test_data = pd.read_csv("../input/test_V2.csv")#, nrows=100000)
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
# test_features 
# train_data.columns
%%time

model_test = RandomForestRegressor(bootstrap = True, max_depth = 50, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 5, random_state=42, n_jobs=-1)
# model_test = RandomForestRegressor( \
#                                    \ #n_estimators = 2000,\
#                                    random_state=42, n_jobs=-1)
model_test.fit(train_data[features], target)
test_pred = model_test.predict(test_data[features])
submission = pd.DataFrame({'Id': test_data['Id'], 'winPlacePerc': list(test_pred)})
submission.to_csv("submission.csv", index=False)
end=time.time()
print((end-start)/60)
