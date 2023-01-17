# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 10)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/train.csv', header=0)
test_data=pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/test.csv', header=0)
train_data.head()
test_data.head()
train_data.info()
# Standard ML Models for comparison

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# Splitting data into training/testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

# Distributions
import scipy
train_data.describe()
numerical_columns=['id',
 'Age',
 'Driving_License',
 'Region_Code',
 'Previously_Insured',
 'Annual_Premium',
 'Policy_Sales_Channel',
 'Vintage']


caterogical_columns=['Gender','Vehicle_Age','Vehicle_Damage']

train_data["Gender"].replace({"Male":0, "Female":1}, inplace=True)

test_data["Gender"].replace({"Male":0, "Female":1}, inplace=True)
train_data["Vehicle_Age"].replace({"< 1 Year": 0, "1-2 Year":1, "> 2 Years":3}, inplace=True)

test_data["Vehicle_Age"].replace({"< 1 Year": 0, "1-2 Year":1, "> 2 Years":3}, inplace=True)
train_data["Vehicle_Damage"].replace({"Yes": 0, "No":1}, inplace=True)

test_data["Vehicle_Damage"].replace({"Yes": 0, "No":1}, inplace=True)
train_data['Gender'].unique()
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr())
%matplotlib inline
import matplotlib.pyplot as plt
train_data.hist(bins=50, figsize=(20,15))
plt.show()
from pandas.plotting import scatter_matrix
scatter_matrix(train_data, figsize=(20, 15))
for each in train_data.columns.to_list():
    #print(len(train_data[each].unique()),each)
    if len(train_data[each].unique())<40:
        carrier_count = train_data[each].value_counts()
        sns.set(style="darkgrid")
        sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
        plt.title('Frequency Distribution of Carriers')
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel(each, fontsize=12)
        plt.figure(figsize=(60,24))
        plt.show()
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="most_frequent")


from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
('imputer', SimpleImputer(strategy="most_frequent")),
('std_scaler', StandardScaler()),
])



from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()

num_attribs = numerical_columns
cat_attribs = caterogical_columns




full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", OneHotEncoder(), cat_attribs),
])
train_data_prepared = full_pipeline.fit_transform(train_data)

test_data_prepared = full_pipeline.fit_transform(test_data)

valid_fraction = 0.05
valid_size = int(len(train_data) * valid_fraction)

train = train_data[:-2 * valid_size]
valid = train_data[-2 * valid_size:-valid_size]
test = train_data[-valid_size:]


# train_l = train_data[:-2 * valid_size]
# valid_l = train_data[-2 * valid_size:-valid_size]
# test_l = train_data[-valid_size:]
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
import seaborn as sns
from collections import Counter

feature_cols = train.columns.drop('Response')


params = {}
params['learning_rate'] = 0.045
params['max_depth'] = 18
params['n_estimators'] = 3000
params['objective'] = 'binary'
params['boosting_type'] = 'gbdt'
params['subsample'] = 0.7
params['random_state'] = 42
params['colsample_bytree']=0.7
params['min_data_in_leaf'] = 55
params['reg_alpha'] = 1.7
params['reg_lambda'] = 1.11
params['class_weight']: {0: 0.5, 1: 0.5}



clf = lgb.LGBMClassifier(**params)
clf.fit(train[feature_cols], train['Response'], early_stopping_rounds=100, eval_set=[(valid[feature_cols], valid['Response']),
        (test[feature_cols], test['Response'])], eval_metric='multi_error', verbose=True)

eval_score = roc_auc_score(train_data['Response'], clf.predict(train_data[feature_cols]))

print('Eval ACC: {}'.format(eval_score))
best_iter = clf.best_iteration_
params['n_estimators'] = best_iter
print(params)


clf = lgb.LGBMClassifier(**params)

clf.fit(train_data[feature_cols], train_data['Response'], eval_metric='multi_error', verbose=False)


eval_score_acc = roc_auc_score(train_data['Response'], clf.predict(train_data[feature_cols]))

print('ACC: {}'.format(eval_score_acc))
preds = clf.predict(test_data)

Counter(preds)
submission = pd.DataFrame({'id':test_data['id'], 'Response':preds})


plt.rcParams['figure.figsize'] = (12, 6)
lgb.plot_importance(clf)
plt.show()
submission.to_csv('/kaggle/working/submission.csv', index=False)
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = SVR()#RandomizedSearchCV(SVR(), param_distributions, n_iter=10, verbose=2, cv=3, random_state=42)
rnd_search_cv.fit(train_data_prepared, train_data['Response'])
rnd_search_cv.best_estimator_
y_pred = rnd_search_cv.best_estimator_.predict(train_data_prepared)
score = roc_auc_score(train_data_prepared, train_data['Response'])
score
preds = rnd_search_cv.best_estimator_.predict(train_data_prepared)

Counter(preds)
submission = pd.DataFrame({'id':test_data['id'], 'Response':preds})

submission.to_csv('/kaggle/working/submission_svm.csv', index=False)
train_data.info()
x=train_data.drop(columns={'id','Response'},axis=1)
y=train_data.loc[:,['Response']]
test=test_data.drop(columns={'id'},axis=1)
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
from sklearn.metrics import accuracy_score,roc_auc_score
pd.set_option('display.max_columns', 100)
import warnings
warnings.filterwarnings("ignore")
import time
test.shape
%%time
err = [] 
y_pred_tot_lgm = np.zeros((len(test), 2))


fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
i = 1

for train_index, test_index in fold.split(x, y):
    x_train, x_val = x.iloc[train_index], x.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    m = CatBoostClassifier(n_estimators=10000,
                       random_state=2020,
                       eval_metric='Accuracy',
                       learning_rate=0.08,
                       depth=8,
                       bagging_temperature=0.3,
                       task_type='GPU'
                       #num_leaves=64
                       
                       )
    m.fit(x_train, y_train,eval_set=[(x_val, y_val)], early_stopping_rounds=100,verbose=200)
    pred_y = m.predict(x_val)
    print(i, " err_lgm: ", accuracy_score(y_val,pred_y))
    err.append(roc_auc_score(y_val,pred_y))
    y_pred_tot_lgm+= m.predict_proba(test)
    i = i + 1
y_pred_tot_lgm=y_pred_tot_lgm/10
sum(err)/10

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

C = 5
alpha = 1 / (C * len(x))

lin_clf = LinearSVC(loss="hinge", C=C, random_state=42)
svm_clf = SVC(kernel="linear", C=C)
sgd_clf = SGDClassifier(loss="hinge", learning_rate="constant", eta0=0.001, alpha=alpha,
                        max_iter=1000, tol=1e-3, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

lin_clf.fit(x, y)
svm_clf.fit(x, y)
sgd_clf.fit(x, y)

