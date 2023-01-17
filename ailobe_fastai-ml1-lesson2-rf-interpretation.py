%%time
# times the whole cell

# import pandas
import pandas as pd

# set the path to read df_raw
path = '../input/df_raw'

# read the data into a pandas DataFrame
df_raw = pd.read_feather(path)
# import proc_df and the functions it depends on
from fastai.structured import numericalize, fix_missing, proc_df

X, y, nas_dict = proc_df(df_raw, 'SalePrice')
# create a function for splitting X and y into train and test sets of customizable sizes
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

# validation set size: 12000.
validation = 12000

# split point: length of dataset minus validation set size.
split_point = len(X)-validation

# split X
X_train, X_valid = split_vals(X, split_point)

# split y
y_train, y_valid = split_vals(y, split_point)

# dimensions (row, columns) of X_train, y_train and X_valid
X_train.shape, y_train.shape, X_valid.shape
# import numpy
import numpy as np

# create a function that takes the RMSE
def rmse(pred,known): return np.sqrt(((pred-known)**2).mean())

# create a function that rounds to 5 decimal places (like kaggle leaderboard)
def rounded(value): return np.round(value, 5)

# create a function that prints a list of 4 scores, rounded:
# [RMSE of X_train, RMSE of X_valid, R Squared of X_train, R Squared of X_valid]
def print_scores(model):
    RMSE_train = rmse(model.predict(X_train), y_train)
    RMSE_valid = rmse(model.predict(X_valid), y_valid)
    R2_train = model.score(X_train, y_train)
    R2_valid = model.score(X_valid, y_valid)
    scores = [rounded(RMSE_train), rounded(RMSE_valid), rounded(R2_train), rounded(R2_valid)]
    if hasattr(m, 'oob_score_'): scores.append(m.oob_score_) # appends OOB score (if any) to the list 
    print(scores)
# print 5 first rows
df_raw.head()
# import set_rf_samples
from fastai.structured import set_rf_samples

# set random subsample size to 50,000 rows
set_rf_samples(50000)
# import the class
from sklearn.ensemble import RandomForestRegressor

# instantiate the model with the parameters of the last lesson
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True,
                          random_state=17190)

# fit the model with data and calculate the running time
%time m.fit(X_train, y_train)

# [RMSE of X_train, RMSE of X_valid, R Squared of X_train, R Squared of X_valid, OOB score]
print_scores(m)
# use a list comprehension to loop through the random forest and concatenates the predictions of each individual tree on a new axis
preds = np.stack([tree.predict(X_valid) for tree in m.estimators_])

# dimensions (rows, columns)
preds.shape
# split df_raw and conserve the part of the validation set
_, raw_valid = split_vals(df_raw, split_point)

# make a copy
validation = raw_valid.copy()

# add new column: calculated standard deviation over row axis
validation['pred_std'] = np.std(preds, axis=0)

# add new column: calculated mean over row axis
validation['pred'] = np.mean(preds, axis=0)
# plots the counts of the unique values of YearMade in the validation set
validation.YearMade.value_counts(dropna=False).plot.bar(figsize=(15,4))
# list of selected columns
columns = ['YearMade', 'SalePrice', 'pred', 'pred_std']

# dataframe of selected columns with the rows grouped by the values in YearMade [index 0]
# and with the calculated mean of 'SalePrice', 'pred', 'pred_std'
year = validation[columns].groupby(columns[0]).mean()

# dimensions (rows, columns)
print(year.shape)

# 10 first rows sorted descendingly
year.sort_values(by=['pred_std'],ascending=False).head(10)
# plots the counts of the unique values of ProductSize in the validation set
validation.ProductSize.value_counts(dropna=False).plot.barh()
# list of selected columns
columns = ['ProductSize', 'SalePrice', 'pred', 'pred_std']

# dataframe of selected columns with the rows grouped by the values in ProductSize [index 0]
# and with the calculated mean of 'SalePrice', 'pred', 'pred_std'
size = validation[columns].groupby(columns[0]).mean()

# calculates the ratio between mean standard deviation and mean prediction
(size.pred_std/size.pred).sort_values(ascending=False)
# import rf_feat_importance
from fastai.structured import rf_feat_importance

# create a dataframe of the feature importance by passing the model and the training set
fi = rf_feat_importance(m, X_train)

# dimension (rows, columns)
print(fi.shape)

# first 5 rows
fi.head()
# plot the feature importance of the column names in 'cols'
fi.plot('cols', 'imp', 'bar', figsize=(15,4))
# create a Series with the column names in 'cols' with a greater 'imp' value than 0.005
to_keep = fi[fi.imp>0.005].cols

# dimensions (rows,)
len(to_keep)
# plot the feature importance of the first 10 columns
fi[:10].plot('cols', 'imp', 'barh', figsize=(15,7))
# create a dataframe with the selected columns
X_keep = X[to_keep].copy()

# split X
X_train, X_valid = split_vals(X_keep, split_point)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)
# create a dataframe of the feature importance
fi = rf_feat_importance(m, X_train)

# plot the feature importance of the first 10 columns 
fi[:10].plot('cols', 'imp', 'barh', figsize=(15,7))
# dtypes in the DataFrame
print(df_raw[['YearMade','ProductSize','Coupler_System','fiProductClassDesc']].dtypes)

# generate descriptive statistics of all columns
df_raw[['YearMade','ProductSize','Coupler_System','fiProductClassDesc']].describe(include='all')
X_one_hot, _, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(X_one_hot, split_point)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)
# create a dataframe of the feature importance
fi = rf_feat_importance(m, X_train)

# plot the feature importance of the first 10 columns
fi[:10].plot('cols', 'imp', 'barh', figsize=(15,7))
# import stats
from scipy import stats

# create distance matrix
matrix = stats.spearmanr(X_keep).correlation

# show distance matrix as a dataframe for demostration
pd.DataFrame(matrix)
# import average linkage
from scipy.cluster.hierarchy import average

# create linkage matrix
linkage = average(matrix)

# show linkage matrix as a dataframe for demostration
pd.DataFrame(linkage)
# import dendrogram
from scipy.cluster.hierarchy import dendrogram

# import matplotlib
import matplotlib.pyplot as plt

# set the size of the figure (plot container)
plt.figure(figsize=(15,7))

# plot dendrogram (inside figure)
dendrogram(linkage, labels=X_keep.columns, orientation='left', leaf_font_size=16)

# display figure
plt.show()
# create a function that takes a dataframe as argument and returns the oob_score of a random forest trained on that fataframe
def get_oob(dataframe):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, max_features=0.6, n_jobs=-1, oob_score=True)
    X, _ = split_vals(dataframe, split_point)
    m.fit(X, y_train)
    return m.oob_score_
# baseline to compare to
get_oob(X_keep)
# loop through the selected columns and print the oob_score with that column removed from the dataframe
for column in ('saleYear', 'saleElapsed',
               'ProductGroup' ,'ProductGroupDesc',
               'fiBaseModel','fiModelDesc',
               'Grouser_Tracks', 'Coupler_System', 'Hydraulics_Flow'):
    print(column, get_oob(X_keep.drop(column, axis=1)))
# list of columns names
to_drop = ['saleYear', 'ProductGroupDesc', 'fiBaseModel', 'Hydraulics_Flow']

# returns oob_score with the selected columns removed
get_oob(X_keep.drop(to_drop, axis=1))
# drop inplace the selected columns from X_keep
X_keep.drop(to_drop, axis=1, inplace=True)

# split X_keep
X_train, X_valid = split_vals(X_keep, split_point)
# import reset_rf_samples
from fastai.structured import reset_rf_samples

# use full bootstrap sample
reset_rf_samples()
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)
set_rf_samples(50000)
X_one_hot, _, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(X_one_hot, split_point)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
fi = rf_feat_importance(m, X_train)
fi[:10].plot('cols', 'imp', 'barh', figsize=(15,4))
# import get_sample
from fastai.structured import get_sample

# random sample of 1000 rows from X_train
X_sample = get_sample(X_train, 1000)
# import partial dependence calculator and plotter
from pdpbox.pdp import pdp_isolate, pdp_plot

# list of features names
features = ['Enclosure_EROPS', 'Enclosure_EROPS AC', 'Enclosure_EROPS w AC', 'Enclosure_NO ROPS', 'Enclosure_None or Unspecified', 'Enclosure_OROPS']

# calculate partial dependence plot (model, dataframe, dataframe.columns, 'feature') 
pdp = pdp_isolate(m, X_sample, X_sample.columns, features)

# plot partial dependent plot (pdp_isolate, 'name')
pdp_plot(pdp, 'Enclosure')
pd.set_option('display.max_columns', None)
X_train[X_train.YearMade == 1000].head()
# pick only rows that have a YearMade value larger than 1000
X_sample = get_sample(X_train[X_train.YearMade>1000], 1000)

# calculate partial dependence plot (model, dataframe, dataframe.columns, 'feature') 
pdp = pdp_isolate(m, X_sample,X_sample.columns,'YearMade')

# plot partial dependent plot (pdp_isolate, 'name')
pdp_plot(pdp, 'YearMade')
# make a copy
X = X_keep.copy()

# create a new target column (empty)
X['in_validation_set'] = None

# set rows in the training set to False (up to last 12000 rows)
X.in_validation_set[:split_point] = False

# set rows in the validation set to True (last 12000 rows)
X.in_validation_set[split_point:] = True

# split X, y
X, y, nas = proc_df(X, 'in_validation_set')
# import the class
from sklearn.ensemble import RandomForestClassifier

m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X, y)
m.oob_score_ # if the validation set was random (not time dependent), this experiment would not work and the oob error would be bad (far from 1).
# create a dataframe of the feature importance
fi = rf_feat_importance(m, X)

# plot the feature importance of first 5 columns
fi.head().plot('cols', 'imp', 'barh')
X['SalesID'].plot()
X['saleElapsed'].plot()
X['MachineID'].plot()
# baseline to compare to

X_train, X_valid = split_vals(X_keep, split_point)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)
# loop through the selected columns and print the scores of a random forest trained with that column removed from the training set.
for column in ('SalesID', 'saleElapsed', 'MachineID'):
    X_train, X_valid = split_vals(X_keep.drop(column, axis=1), split_point)
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    print(column)
    print_scores(m)
# drop the columns from the dataframe and split X
X_train, X_valid = split_vals(X_keep.drop(['SalesID', 'MachineID'], axis=1), split_point)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)
# use full bootstrap sample
reset_rf_samples()
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)
# plot the feature importance of the column names in 'cols'
rf_feat_importance(m, X_train).plot('cols', 'imp', 'barh', figsize=(12,7))
m = RandomForestRegressor(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_scores(m)