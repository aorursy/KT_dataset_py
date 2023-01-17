#import matplotlib.pyplot as plt

#import plotly.express as px

from tqdm import tqdm

#from sklearn.preprocessing import StandardScaler

#from sklearn.svm import NuSVR

#from sklearn.metrics import mean_absolute_error

import pandas as pd

import numpy as np

import seaborn as sns 

from sklearn import tree

from sklearn.model_selection import GridSearchCV 
train = pd.read_csv('../input/LANL-Earthquake-Prediction/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
pd.options.display.precision = 15

train.head(3)
train.shape
x = train[['acoustic_data']].iloc[0:15000, 0:]

ax = sns.distplot(x)
rows = 150000

segments = int(np.floor(train.shape[0] / rows))

print(segments)
X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['mean', 'des', 'std', 'max', 'min', 'quan0.25', 'quan0.5', 'quan0.75'])

print(X_train.shape)



y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



print(y_train.shape)

for segment in tqdm(range(segments)):

    

    seg = train.iloc[segment*rows:segment*rows+rows] 

    

    x = seg['acoustic_data'].values 

    y = seg['time_to_failure'].values[-1] 

    

    

    y_train.loc[segment, 'time_to_failure'] = y

    

    

    X_train.loc[segment, 'mean'] = x.mean()

    X_train.loc[segment, 'des'] = np.var(x)

    X_train.loc[segment, 'std'] = x.std()

    X_train.loc[segment, 'max'] = x.max()

    X_train.loc[segment, 'min'] = x.min()



    X_train.loc[segment, 'quan0.25'] = np.quantile(x, 0.25)

    X_train.loc[segment, 'quan0.5'] = np.quantile(x, 0.5)

    X_train.loc[segment, 'quan0.75'] = np.quantile(x, 0.75)

X_train.head()
y_train.head()
print(X_train.shape)

print(y_train.shape)
reg = tree.DecisionTreeRegressor()
parametrs_des_regr = {'criterion': ['mae', 'friedman_mse'], 'max_depth': range(2, 10), 'min_samples_split': range(59,70), 

             'min_samples_leaf': range(10,20)} 
grid_search_cv_clf = GridSearchCV(reg, parametrs_des_regr, cv=5)

grid_search_cv_clf.fit(X_train, y_train)

print(grid_search_cv_clf.best_params_)

best_clf = grid_search_cv_clf.best_estimator_

best_clf.fit(X_train, y_train)
submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
submission.head()
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.seg_id)
X_test.head()
for seg_id in X_test.index:

    

    seg = pd.read_csv('../input/LANL-Earthquake-Prediction/test/' + seg_id + '.csv')

    

    y = seg['acoustic_data'].values

    

    X_test.loc[seg_id, 'mean'] = y.mean()

    X_test.loc[seg_id, 'des'] = np.var(y)

    X_test.loc[seg_id, 'std'] = y.std()

    X_test.loc[seg_id, 'max'] = y.max()

    X_test.loc[seg_id, 'min'] = y.min()



    X_test.loc[seg_id, 'quan0.25'] = np.quantile(y, 0.25)

    X_test.loc[seg_id, 'quan0.5'] = np.quantile(y, 0.5)

    X_test.loc[seg_id, 'quan0.75'] = np.quantile(y, 0.75)
X_test.head()
predict_time_to_failure = best_clf.predict(X_test)
predict_time_to_failure
submission.head()
submission_pred = pd.DataFrame(columns=submission.columns, dtype=np.float64)
submission_pred.head()
submission_pred['time_to_failure'] = predict_time_to_failure

submission_pred['seg_id'] = submission.seg_id
submission_pred.head()
submission.to_csv('submission1.csv', index=False)