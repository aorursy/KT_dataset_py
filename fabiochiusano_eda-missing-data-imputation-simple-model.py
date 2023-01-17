import cufflinks as cf

cf.set_config_file(offline=True)



import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd 

import os



from pandas_summary import DataFrameSummary



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
local_path = "./data/"

kaggle_path = "/kaggle/input/killer-shrimp-invasion/"

example_submission_filename = "temperature_submission.csv"

train_filename = "train.csv"

test_filename = "test.csv"



base_path = kaggle_path



temperature_submission = pd.read_csv(base_path + example_submission_filename)

test = pd.read_csv(base_path + test_filename)

train = pd.read_csv(base_path + train_filename)
train = train.drop("pointid", axis=1)

test = test.drop("pointid", axis=1)
train.head(5)
len(train), len(test)
train_summary = DataFrameSummary(train)

test_summary = DataFrameSummary(test)
train_summary.columns_stats
train_summary["Salinity_today"]
train_summary["Temperature_today"]
train_summary["Substrate"]
train_summary["Depth"]
train_summary["Exposure"]
train_summary["Presence"]
import missingno as msno

%matplotlib inline
msno.bar(train)
msno.matrix(train)
msno.heatmap(train)
train_without_na = train.dropna()

len(train_without_na[train_without_na["Presence"] == 1]), len(train[train["Presence"] == 1])
train[train["Presence"] == 1]
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



imputer = IterativeImputer(max_iter=10)

imputer.fit(train.drop(labels=["Presence"], axis=1))

new_columns = train.drop(labels=["Presence"], axis=1).columns.tolist()
def impute_missing_data(df, target, imputer, new_columns):

    df_without_target = df.drop(labels=[target], axis=1) if target in df.columns.tolist() else df

    arr_imputed = imputer.transform(df_without_target)

    df_imputed = pd.DataFrame.from_records(arr_imputed, columns=new_columns)

    if target in df.columns.tolist():

        df_imputed["Presence"] = df["Presence"]

    return df_imputed
target = "Presence"

train_imputed = impute_missing_data(train, target, imputer, new_columns)

test_imputed = impute_missing_data(test, target, imputer, new_columns)
train_imputed[train_imputed["Presence"] == 1]
corr = train_imputed.corr()

corr.style.background_gradient(cmap='coolwarm', axis=None)
columns_to_bin = ["Salinity_today", 'Temperature_today', 'Depth', 'Exposure']

for col in columns_to_bin:

    num_bins = 10

    if col == "Depth":

        num_bins = [-1.2**i + 0.5 for i in range(40)][::-1]

    train_imputed[col + '_binned'] = pd.cut(train_imputed[col], num_bins)
pt = train_imputed.pivot_table(columns="Salinity_today_binned", values="Presence", aggfunc='count')

pt.iplot(kind='bar', title="Number of records per salinity bin", xTitle="Salinity", yTitle="Number of records")
pt = train_imputed.pivot_table(columns="Salinity_today_binned", values="Presence", aggfunc=np.count_nonzero)

pt.iplot(kind='bar', title="Number of positive presence per salinity bin", xTitle="Salinity", yTitle="Positive presence")
pt = train_imputed.pivot_table(columns="Salinity_today_binned", values="Presence", aggfunc=np.mean)

pt.iplot(kind='bar', title="Average presence per salinity bin", xTitle="Salinity", yTitle="Average presence")
pt = train_imputed.pivot_table(columns="Temperature_today_binned", values="Presence", aggfunc='count')

pt.iplot(kind='bar', title="Number of records per temperature bin", xTitle="Temperature", yTitle="Number of records")
pt = train_imputed.pivot_table(columns="Temperature_today_binned", values="Presence", aggfunc=np.count_nonzero)

pt.iplot(kind='bar', title="Number of positive presence per temperature bin", xTitle="Temperature", yTitle="Positive presence")
# average presence per temperature bin

pt = train_imputed.pivot_table(columns="Temperature_today_binned", values="Presence", aggfunc=np.mean)

pt.iplot(kind='bar', title="Average presence per temperature bin", xTitle="Temperature", yTitle="Average presence")
pt = train_imputed.pivot_table(columns="Depth_binned", values="Presence", aggfunc='count')

pt.iplot(kind='bar', title="Number of records per depth bin", xTitle="Depth", yTitle="Number of records")
pt = train_imputed.pivot_table(columns="Depth_binned", values="Presence", aggfunc=np.count_nonzero)

pt.iplot(kind='bar', title="Number of positive presence per depth bin", xTitle="Depth", yTitle="Positive presence")
pt = train_imputed.pivot_table(columns="Depth_binned", values="Presence", aggfunc=np.mean)

pt.iplot(kind='bar', title="Number of positive presence per depth bin", xTitle="Depth", yTitle="Positive presence")
pt = train_imputed.pivot_table(columns="Exposure_binned", values="Presence", aggfunc='count')

pt.iplot(kind='bar', title="Number of records per exposure bin", xTitle="Exposure", yTitle="Number of records")
pt = train_imputed.pivot_table(columns="Exposure_binned", values="Presence", aggfunc=np.count_nonzero)

pt.iplot(kind='bar', title="Number of positive presence per exposure bin", xTitle="Exposure", yTitle="Positive presence")
# average presence per temperature bin

pt = train_imputed.pivot_table(columns="Exposure_binned", values="Presence", aggfunc=np.mean)

pt.iplot(kind='bar', title="Average presence per exposure bin", xTitle="Exposure", yTitle="Average presence")
pt = train_imputed.pivot_table(columns="Substrate", values="Presence", aggfunc='count')

pt.iplot(kind='bar', title="Number of records per substrate", xTitle="Substrate", yTitle="Number of records")
pt = train_imputed.pivot_table(columns="Substrate", values="Presence", aggfunc=np.count_nonzero)

pt.iplot(kind='bar', title="Number of positive presence per substrate", xTitle="Substrate", yTitle="Positive presence")
# average presence per temperature bin

pt = train_imputed.pivot_table(columns="Substrate", values="Presence", aggfunc=np.mean)

pt.iplot(kind='bar', title="Average presence per substrate", xTitle="Substrate", yTitle="Average presence")
target = "Presence"

feature_names_no_imputation = ["Salinity_today", "Temperature_today", "Substrate", "Depth", "Exposure"]

feature_names = new_columns
# dataset without imputation

train_fill_na = train.fillna(method='ffill')

X_train_raw = train_fill_na[feature_names_no_imputation]

y_train_raw = train_fill_na[target].astype('category')
# dataset with imputation

X_train_imputed = train_imputed[feature_names]

y_train_imputed = train_imputed[target].astype('category')
# dataset z-scored

import copy

from sklearn.preprocessing import StandardScaler



X_train_imputed_std = copy.deepcopy(X_train_imputed)

test_imputed_std = copy.deepcopy(test_imputed)

columns_to_standardise = ['Salinity_today', 'Temperature_today', 'Depth', 'Exposure']

for col in columns_to_standardise:

    scaler = StandardScaler()

    scaler.fit(train[[col]])

    X_train_imputed_std[col] = scaler.transform(X_train_imputed[[col]])

    test_imputed_std[col] = scaler.transform(test_imputed_std[[col]])

y_train_imputed_std = y_train_imputed
n_splits = 10
from sklearn.dummy import DummyClassifier

from sklearn.model_selection import cross_val_score
classifier = DummyClassifier(strategy="constant", constant=0)



scores = cross_val_score(classifier, X_train_raw, y_train_raw,

                         scoring='roc_auc', cv=n_splits)



scores, scores.mean()
from sklearn.ensemble import ExtraTreesClassifier
classifier = ExtraTreesClassifier(n_estimators=10, n_jobs=4)



scores = cross_val_score(classifier, X_train_raw, y_train_raw,

                         scoring='roc_auc', cv=n_splits)



scores, scores.mean()
classifier = ExtraTreesClassifier(n_estimators=10, n_jobs=4)



scores = cross_val_score(classifier, X_train_imputed, y_train_imputed,

                         scoring='roc_auc', cv=n_splits)



scores, scores.mean()
from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression(solver="lbfgs")



scores = cross_val_score(classifier, X_train_imputed_std, y_train_imputed_std,

                         scoring='roc_auc', cv=n_splits)



scores, scores.mean()
classifier = LogisticRegression().fit(X_train_imputed_std, y_train_imputed_std)
predictions = classifier.predict_proba(test_imputed_std[feature_names])
temperature_submission['Presence'] = 1 - predictions

temperature_submission.head(5)
temperature_submission.to_csv('my_sub.csv', index=False)