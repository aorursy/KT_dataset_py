import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import time

import datetime

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression

import gc

import warnings

warnings.filterwarnings("ignore")

from scipy.signal import convolve

from scipy import stats



import shap
%%time

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
# Split train data into Dataframe containing rows of 150000 data points.



rows = 150000

segments = int(np.floor(train.shape[0] / rows))

X_train = pd.DataFrame(index=range(4194), dtype=np.float64, columns = range(rows))

y_tr = pd.DataFrame(index=range(4194), dtype=np.float64, columns=range(1))



for segment in tqdm_notebook(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    X_train.iloc[segment] = seg['acoustic_data'].values

    y_tr.iloc[segment] = seg['time_to_failure'].values[-1]

    

X_train.head()
y_tr.head()
# Dataframe to create features from test data



X_tr = pd.DataFrame(index=range(4194), dtype=np.float64)



for segment in tqdm_notebook(X_train.index):

    X_tr.loc[segment, 'mean'] = X_train.loc[segment].mean()

    X_tr.loc[segment,'max'] = X_train.loc[segment].max()

    X_tr.loc[segment,'min'] = X_train.loc[segment].min()

    X_tr.loc[segment, 'std'] = X_train.loc[segment].std()

    X_tr.loc[segment, 'var'] = X_train.loc[segment].var()

    X_tr.loc[segment, 'kurt'] = X_train.loc[segment].kurtosis()

    X_tr.loc[segment, 'skew'] = X_train.loc[segment].skew()

    X_tr.loc[segment, 'med'] = X_train.loc[segment].median()

    X_tr.loc[segment, 'sum'] = X_train.loc[segment].sum()

    X_tr.loc[segment, 'q91'] = np.quantile(X_train.loc[segment], 0.91)

    X_tr.loc[segment, 'q92'] = np.quantile(X_train.loc[segment], 0.92)

    X_tr.loc[segment, 'q93'] = np.quantile(X_train.loc[segment], 0.93)

    X_tr.loc[segment, 'q94'] = np.quantile(X_train.loc[segment], 0.94)

    X_tr.loc[segment, 'q95'] = np.quantile(X_train.loc[segment], 0.95)

    X_tr.loc[segment, 'q96'] = np.quantile(X_train.loc[segment], 0.96)

    X_tr.loc[segment, 'q97'] = np.quantile(X_train.loc[segment], 0.97)

    X_tr.loc[segment, 'q98'] = np.quantile(X_train.loc[segment], 0.98)

    X_tr.loc[segment, 'q99'] = np.quantile(X_train.loc[segment], 0.99)

    X_tr.loc[segment, 'q01'] = np.quantile(X_train.loc[segment], 0.01)

    X_tr.loc[segment, 'q02'] = np.quantile(X_train.loc[segment], 0.02)

    X_tr.loc[segment, 'q03'] = np.quantile(X_train.loc[segment], 0.03)

    X_tr.loc[segment, 'q04'] = np.quantile(X_train.loc[segment], 0.04)

    X_tr.loc[segment, 'q05'] = np.quantile(X_train.loc[segment], 0.05)

    X_tr.loc[segment, 'q06'] = np.quantile(X_train.loc[segment], 0.06)

    X_tr.loc[segment, 'q07'] = np.quantile(X_train.loc[segment], 0.07)

    X_tr.loc[segment, 'q08'] = np.quantile(X_train.loc[segment], 0.08)

    X_tr.loc[segment, 'q09'] = np.quantile(X_train.loc[segment], 0.09)

    X_tr.loc[segment, 'q999'] = np.quantile(X_train.loc[segment],0.999)

    X_tr.loc[segment, 'q001'] = np.quantile(X_train.loc[segment],0.001)



    

X_tr.head(10)

X_tr.shape
scaler = StandardScaler()

scaler.fit(X_tr)

X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
# Reading test files into a dataframe,

# with rows as file names and columns as the acoustic signal data. 



submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X = pd.DataFrame(columns=range(0,150000), dtype=np.float64, index=submission.index)



for seg_id in tqdm_notebook(X.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    X.loc[seg_id,] = seg['acoustic_data'].values



X.head()
# Remove index name seg_id

del X.index.name

X.head()
X_test = pd.DataFrame(index=X.index, dtype=np.float64)



for row in tqdm_notebook(X.index):

    X_test.loc[row, 'mean'] = X.loc[row].mean()

    X_test.loc[row,'max'] = X.loc[row].max()

    X_test.loc[row,'min'] = X.loc[row].min()

    X_test.loc[row, 'std'] = X.loc[row].std()

    X_test.loc[row, 'var'] = X.loc[row].var()

    X_test.loc[row, 'kurt'] = X.loc[row].kurtosis()

    X_test.loc[row, 'skew'] = X.loc[row].skew()

    X_test.loc[row, 'med'] = X.loc[row].median()

    X_test.loc[row, 'sum'] = X.loc[row].sum()

    X_test.loc[row, 'q91'] = np.quantile(X.loc[row], 0.91)

    X_test.loc[row, 'q92'] = np.quantile(X.loc[row], 0.92)

    X_test.loc[row, 'q93'] = np.quantile(X.loc[row], 0.93)

    X_test.loc[row, 'q94'] = np.quantile(X.loc[row], 0.94)

    X_test.loc[row, 'q95'] = np.quantile(X.loc[row], 0.95)

    X_test.loc[row, 'q96'] = np.quantile(X.loc[row], 0.96)

    X_test.loc[row, 'q97'] = np.quantile(X.loc[row], 0.97)

    X_test.loc[row, 'q98'] = np.quantile(X.loc[row], 0.98)

    X_test.loc[row, 'q99'] = np.quantile(X.loc[row], 0.99)

    X_test.loc[row, 'q01'] = np.quantile(X.loc[row], 0.01)

    X_test.loc[row, 'q02'] = np.quantile(X.loc[row], 0.02)

    X_test.loc[row, 'q03'] = np.quantile(X.loc[row], 0.03)

    X_test.loc[row, 'q04'] = np.quantile(X.loc[row], 0.04)

    X_test.loc[row, 'q05'] = np.quantile(X.loc[row], 0.05)

    X_test.loc[row, 'q06'] = np.quantile(X.loc[row], 0.06)

    X_test.loc[row, 'q07'] = np.quantile(X.loc[row], 0.07)

    X_test.loc[row, 'q08'] = np.quantile(X.loc[row], 0.08)

    X_test.loc[row, 'q09'] = np.quantile(X.loc[row], 0.09)

    X_test.loc[row, 'q999'] = np.quantile(X.loc[row],0.999)

    X_test.loc[row, 'q001'] = np.quantile(X.loc[row],0.001)



X_test.head()
X_test.shape
scaler = StandardScaler()

scaler.fit(X_test)

X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
# RF regressor model



from sklearn.ensemble import RandomForestRegressor



forest = RandomForestRegressor(n_estimators=1000, max_depth = 100, min_samples_leaf = 30, min_samples_split = 30)

forest.fit(X_train_scaled.values, y_tr.values)



#print(regr_lanl.feature_importances_)



prediction_rf = forest.predict(X_test_scaled.values)



submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

submission['time_to_failure'] = prediction_rf

print(submission.head())

submission.to_csv('submission_rf.csv')
# print the JS visualization code to the notebook

shap.initjs()



#X1,y = shap.datasets.adult()

#X_display,y_display = shap.datasets.adult(display=True)
explainer = shap.TreeExplainer(forest)

shap_values = explainer.shap_values(X_train_scaled)
shap.force_plot(explainer.expected_value, shap_values[0,:], X_train_scaled.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[4,:], X_train_scaled.iloc[4,:])

shap.summary_plot(shap_values,X_train_scaled)
shap.summary_plot(shap_values, X_train_scaled, plot_type="bar")
for name in X_tr.columns:

    shap.dependence_plot(name, shap_values, X_train_scaled)