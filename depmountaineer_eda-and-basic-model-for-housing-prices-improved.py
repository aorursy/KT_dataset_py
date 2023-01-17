import pandas as pd
x=pd;print(x.__name__, "version:", x.__version__)
import numpy as np
x=np;print(x.__name__, "version:", x.__version__)
import scipy
x=scipy;print(x.__name__, "version:", x.__version__)
import sklearn
x=sklearn;print(x.__name__, "version:", x.__version__)
import xgboost as xgb
x=xgb;print(x.__name__, "version:", x.__version__)
import lightgbm as lgb
x=lgb;print(x.__name__, "version:", x.__version__)
import keras
x=keras;print(x.__name__, "version:", x.__version__)
import tensorflow as tf
x=tf;print(x.__name__, "version:", x.__version__)
import torch
x=torch;print(x.__name__, "version:", x.__version__)
import matplotlib
import matplotlib.pyplot as plt
x=matplotlib;print(x.__name__, "version:", x.__version__)
import seaborn as sns
x=sns;print(x.__name__, "version:", x.__version__)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import os, sys, math, datetime, shutil, pickle
from IPython.core.interactiveshell import InteractiveShell
if not os.path.isfile("../working/data_tools.py"):
    shutil.copyfile(src = "../input/python-modules/data_tools.py", dst = "../working/data_tools.py")
import data_tools as dt
InteractiveShell.ast_node_interactivity = "all"
%matplotlib inline
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10)
plt.style.use('seaborn-poster')
sns.set(palette='deep')
sns.set(context='poster')
ScriptName = "EDA HousePrices"
ScriptFile = "eda_houseprices.ipynb"
ScriptDescription = "A Jupyter notebook for EDA and basic prediction of house prices"

run_first = []

analysis = dt.Analysis(inputs='../input/house-prices-advanced-regression-techniques')
analysis

#Script Input Variables
infiles = analysis.inputs

infile_train = analysis.train
infile_test =  analysis.test
outfile_final = analysis.final_output
index = analysis.ids
target = analysis.target

model = Ridge(random_state=0, alpha=0.01)

also_predict = True
train_validation_ratio = 5

{os.path.splitext(x)[0]:'../input/house-prices-advanced-regression-techniques' + x for x in os.listdir('../input/house-prices-advanced-regression-techniques') if os.path.splitext(x)[1] == '.csv' and x != 'sample_submission.csv'}
# if not os.path.isfile(ScriptFile):
#     raise FileNotFoundError(ScriptFile)
# for file in run_first:
#     if not os.path.isfile(file):
#         raise FileNotFoundError(file)
if not os.path.isfile(infile_train):
    raise FileNotFoundError(infile_train)
if not os.path.isfile(infile_test):
    raise FileNotFoundError(infile_test)
print("Running:", ScriptName, "in file:", ScriptFile)
print("Inputs:", infile_train, infile_test)
print("Final Output:", outfile_final)
print("Model:", model)
df_train = dt.load_df(infile_train, index=index, name='train')
features = list(df_train.columns)
features.remove(target)
df_train.head()
found = 0
unique = df_train.nunique()
for n in [x for x in unique.unique() if len(unique[unique==x]) > 1]:
    idx = unique[unique==n].index
    unique_same_size = df_train[idx].apply(pd.factorize)
    for col1 in idx:
        for col2 in idx:
            if col2 != col1:
                if ((unique_same_size[col2][0]==unique_same_size[col1][0]).all()):
                    print('col: ', col2, 'equals col:', col1)
                    found += 1
if found:
    print("%s Equal factorized columns found" % found)
else:
    print("All columns are distinct")
if(1 in unique):
    print("Some columns have just one unique value: %s" % unique[unique==1])
else:
    print("No columns have just one unique value")
df_test = dt.load_df(infile_test, index=index, name='test')
df_test.head()
if set(df_test.columns) == set(features):
    print("Train and test sets have same features")
else:
    if set(df_test.columns) - set(features):
        print("Features in test set that are not in train set:", set(df_test.columns) - set(features))
    if set(features) - set(df_test.columns):
        print("Features in test set that are not in train set:", set(df_test.columns) - set(features))
pvalues = pd.DataFrame(columns=['p'])
pvalues.index.name='Feature'
for col in features:
    if df_train[col].dtype != np.object:
        d, p = scipy.stats.ks_2samp(df_train[col].values, df_test[col].values)
        pvalues.loc[col,'p'] = p
    else:#factorize categorical features over all train/test data
        l = len(df_train[col])
        fl = pd.factorize(list(df_train[col]) + list(df_test[col]))
        d, p = scipy.stats.ks_2samp(fl[0][:l], fl[0][l:])
        pvalues.loc[col, 'p'] = p        
print("Columns in which train distribution differs significantly from test distribution at an alpha of 5%")
significant = pvalues[pvalues.p <= 0.05]
significant
if(len(significant) == 0):
    print("Test and Train data likely are identically distributed")
one_hot_features = [x for x in df_train.columns if np.dtype(df_train[x])==np.object and df_train[x].nunique() > 2]
col1=dt.one_hot(df=df_train, columns=one_hot_features, abbr=True)
col2=dt.one_hot(df=df_test, columns=one_hot_features, abbr=True)
features = list(df_test.columns)
df_ftrain = df_train.copy()
df_ftest = df_test.copy()
for col in features:
    l = len(df_train[col])
    if df_ftrain[col].dtype == np.object:
        fl = pd.factorize(list(df_train[col]) + list(df_test[col]))[0]
        df_ftrain[col] = fl[:l]
        df_ftest[col] = fl[l:]
df_ftrain = df_ftrain.reindex_axis(df_ftrain.corrwith(other=df_ftrain[target]).abs().sort_values(ascending=False).index, axis=1)
corr = df_ftrain.corr()
sns.heatmap(corr)
corr
best_features = [x for x in corr.columns if np.abs(corr.loc[target, x]) >= 0.1]
if target in best_features:
    best_features.remove(target)
best_features
df_ftrain.fillna(df_ftrain.mean(), inplace=True)
df_ftest.fillna(df_ftrain.mean(), inplace=True)
rs = sklearn.model_selection.ShuffleSplit(n_splits=20, random_state=0, train_size=(train_validation_ratio-1.0)/train_validation_ratio, test_size=1.0/train_validation_ratio)
scores = []
for train_idx, test_idx in rs.split(df_train[features], df_train[target]):
    df_vtrain = df_ftrain.iloc[train_idx]
    df_vtest = df_ftrain.iloc[test_idx].copy()
    model = model.fit(df_vtrain[best_features], np.log(1+df_vtrain[target]))
    df_vtest['PredictedSalePrice'] = model.predict(df_vtest[best_features])
    score = mean_squared_error(np.log(1+df_vtest[target]), df_vtest['PredictedSalePrice'])
    scores.append(score)
pd.DataFrame(scores).describe()
score = pd.DataFrame(scores).describe().iloc[6,0]
print("Validation score:", score)
if also_predict:
    model.fit(df_ftrain[best_features], np.log(1+df_ftrain[target]))
if also_predict:
    df_ftest[target] = model.predict(df_ftest[best_features])
if also_predict:
    result = np.exp(df_ftest[[target]])-1
    result.head()
#os.makedirs(os.path.dirname(outfile_final), exist_ok=True)
#Kaggle won't let you write to ../output, so write to $CWD instead.
outfile_final = 'submission.csv'
if also_predict:
    cols = list(analysis.submission_fields)
    if analysis.main_id in cols:
        cols.remove(analysis.main_id)
    dt.validate_submission(df=result, cols=cols, index=analysis.main_id, mins=[0,0], maxes=[100000000, 100000000])
    result.to_csv(outfile_final, index=True)
#useful on home computer, not on Kaggle kernel

# if also_predict:
#     d = '../provisional/'+datetime.datetime.today().strftime("%Y%m%d") + '_' + str(score)
#     try:
#         os.makedirs(d)
#     except FileExistsError:
#         print("Not making directory:", d, "-- directory exists")
#     for f in run_first + [ScriptFile]:
#         try:
#             shutil.copyfile(f, d + '/' + f )
#         except FileExistsError:
#             print("Not copying:", f, 'to:', d + '/' + f,'-- file exists' )
#     f = 'submission.csv'
#     try:
#         shutil.copyfile('../output/' + f, d + '/' + f )
#     except FileExistsError:
#         print("Not copying ../output/" + f, 'to:', d + '/' + f,'-- file exists' )
print("Validation score:", score)
try:
    best_score = pickle.load(open('best_score.p', 'rb'))
    print("Previous best:   ", best_score)
    if score < best_score:
        print("New best! Improved by: %s%%" % (math.floor(10000.0*(best_score - score)/best_score)/100.0,))
        best_score = score
        pickle.dump(best_score, open('best_score.p', 'wb'))
    elif score == best_score:
        print("Score equals previous best")
    else:
        print("Score is worse than previous best")
except FileNotFoundError:
    best_score = score
    pickle.dump(best_score, open('best_score.p', 'wb'))