# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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
import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv(filepath_or_buffer='/kaggle/input/train_LZdllcl.csv')

test = pd.read_csv(filepath_or_buffer='/kaggle/input/test_2umaH9m.csv')
train.shape, test.shape
train.duplicated().sum(), test.duplicated().sum()
train.head()
train.info()
train.nunique()
train.is_promoted.value_counts(), train.is_promoted.value_counts(normalize=True) 
train.drop(columns='employee_id', inplace=True)

test.drop(columns='employee_id', inplace=True)
for col in ['department', 'region', 'education', 'gender', 'recruitment_channel', 

            'previous_year_rating', 'KPIs_met >80%', 'awards_won?']:

    train[col] = train[col].astype('object')

    test[col] = test[col].astype('object')
train.head()
train.education.fillna(train.education.mode()[0], inplace=True)

train.previous_year_rating.fillna(train.previous_year_rating.median(), inplace=True)
# encoder cannot handle Nan

# function to encode non-null data and replace it in the original data

def encode(data):

    from sklearn.preprocessing import OrdinalEncoder

    encoder = OrdinalEncoder()   

    

    #retains only non-null values

    nonulls = np.array(data.dropna())

    #reshapes the data for encoding

    impute_reshape = nonulls.reshape(-1,1)

    #encode data

    impute_ordinal = encoder.fit_transform(impute_reshape)

    #Assign back encoded values to non-null values

    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data
# create a list of categorical columns to iterate over

cat_cols = train.columns[((train.dtypes == 'object') | (train.dtypes == 'category'))]



#create a for loop to iterate through each column in the data

for columns in cat_cols[0: -1]:

    encode(train[columns])

    encode(test[columns])
train.head()
train['education'] = train['education'].astype('float64')

train['previous_year_rating'] = train['previous_year_rating'].astype('float64')

train['awards_won?'] = train['awards_won?'].astype('float64')

test['education'] = test['education'].astype('float64')

test['previous_year_rating'] = test['previous_year_rating'].astype('float64')

test['awards_won?'] = test['awards_won?'].astype('float64')
# from sklearn.impute import KNNImputer

# imputer = KNNImputer()

# encoded_train = pd.DataFrame(np.round(imputer.fit_transform(train)), columns = train.columns)

# test['is_promoted'] = np.nan

# encoded_test = pd.DataFrame(np.round(imputer.transform(test)), columns = test.columns)
# test.drop(columns='is_promoted', inplace=True)

# encoded_test.drop(columns='is_promoted', inplace=True)
X = train.drop(columns='is_promoted')

y = train['is_promoted']

X_test = test



# X = encoded_train.drop(columns='is_promoted')

# y = encoded_train['is_promoted']

# X_test = encoded_test



X.shape, y.shape, X_test.shape
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from lightgbm import LGBMClassifier

model = LGBMClassifier(max_depth=5,

                       learning_rate=0.4, 

                       n_estimators=100)



model.fit(x_train,y_train,

          eval_set=[(x_train,y_train),(x_val, y_val.values)],

          eval_metric='auc',

          early_stopping_rounds=100,

          verbose=200)



pred_y = model.predict_proba(x_val)[:,1]
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

print(roc_auc_score(y_val, pred_y))

confusion_matrix(y_val, pred_y>0.5)
import plotly.express as px

fpr, tpr, thresholds = roc_curve(y_val, pred_y)

fig = px.line(x=fpr, y=tpr, width=400, height=400,

              labels={'x':'False Positive Rates','y':'True Positive Rates'})

fig.show()
import lightgbm

lightgbm.plot_importance(model)
err = []

y_pred_tot_lgm = []



from sklearn.model_selection import StratifiedKFold



fold = StratifiedKFold(n_splits=15)

i = 1

for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = LGBMClassifier(boosting_type='gbdt',

                       max_depth=5,

                       learning_rate=0.05,

                       n_estimators=5000,

                       min_child_weight=0.01,

                       colsample_bytree=0.5,

                       random_state=1994)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          eval_metric='auc',

          verbose=200)

    pred_y = m.predict_proba(x_val)[:,1]

    print("err_lgm: ",roc_auc_score(y_val,pred_y))

    err.append(roc_auc_score(y_val, pred_y))

    pred_test = m.predict_proba(X_test)[:,1]

    i = i + 1

    y_pred_tot_lgm.append(pred_test)
np.mean(err,0)
submission = pd.read_csv(filepath_or_buffer='/kaggle/input/test_2umaH9m.csv')

np.mean(y_pred_tot_lgm, 0).max()
pd.read_csv(filepath_or_buffer='/kaggle/input/test_2umaH9m.csv').shape
len(np.mean(y_pred_tot_lgm, 0))
submission = pd.DataFrame([pd.read_csv(filepath_or_buffer='/kaggle/input/test_2umaH9m.csv').iloc[:, 0], np.mean(y_pred_tot_lgm, 0)])
submission = submission.T

submission['employee_id'] = submission['employee_id'].astype('int')

submission.rename(columns={'Unnamed 0': 'is_promoted'}, inplace=True)

submission['is_promoted'] = submission['is_promoted'].round()

submission['is_promoted'] = submission['is_promoted'].astype('int')



submission.head()
submission.to_csv('submit.csv', index=False, header=True)