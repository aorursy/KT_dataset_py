import numpy as np

from numpy import sort, mean



from collections import Counter

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, roc_auc_score, precision_score, recall_score, matthews_corrcoef, make_scorer

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold, cross_val_score

from sklearn.feature_selection import SelectFromModel



from scipy import stats

from scipy.stats import randint



import xgboost as xgb

from xgboost import plot_importance



%matplotlib inline
import os

import multiprocessing



mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448

mem_gib = mem_bytes/(1024.**3)  # e.g. 3.74

print("RAM: %f GB" % mem_gib)

print("CORES: %d" % multiprocessing.cpu_count())

!lscpu | grep 'Thread(s) per core'
df = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/train.csv', index_col='row_id')

users = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/users.csv')

df = pd.merge(df, users, on='user_id')
df.info()
df.isnull().sum(axis = 0)
df.describe(include='all')
hols = [

    20190706,

    20190707,

    20190713,

    20190714,

    20190720,

    20190721,

    20190727,

    20190728,

    20190803,

    20190804,

    20190809, 

    20190810,

    20190811,

    20190812,

    20190817,

    20190818,

    20190824,

    20190825,

    20190825,

    20190831,

    20190901,

    20190907,

    20190908,

    20190914,

    20190915,

    20190921,

    20190922,

    20190928,

    20190929,

]
def clean_data(df):

    df["day_of_week"] = pd.to_datetime(df["grass_date"]).dt.dayofweek

    df["day_of_week"] = df["day_of_week"].astype('category')

    

    df["quarter_of_month"] = pd.to_datetime(df["grass_date"]).dt.day

    df["quarter_of_month"] = df["quarter_of_month"].apply(lambda x: int((x-1)/8)).astype('category')

    

    df["grass_date"] = pd.to_datetime(df["grass_date"]).dt.strftime("%Y%m%d").astype(int)

    

    df['is_hol'] = [(day in hols) for day in df.grass_date.values]

    df['is_hol'] = df['is_hol'].astype(bool)

    

    df['user_id_len'] = df.user_id.astype(str).str.len()

    df['user_id_len'] = df['user_id_len'].astype('category')



    df.loc[df.last_open_day == 'Never open', 'last_open_day'] = 999

    df.last_open_day = df.last_open_day.astype(int)

    df.last_open_day = df.last_open_day.clip(0, 999)



    df.loc[df.last_login_day == 'Never login', 'last_login_day'] = 1825

    df.last_login_day = df.last_login_day.astype(int)

    df.last_login_day = df.last_login_day.clip(0, 1825)



    df.loc[df.last_checkout_day == 'Never checkout', 'last_checkout_day'] = 1825

    df.last_checkout_day = df.last_checkout_day.astype(int)

    df.last_checkout_day = df.last_checkout_day.clip(0, 1825)



    df.domain = df.domain.astype('category')

    

    df.loc[df.attr_1.isna(), 'attr_1'] = -1

    df.attr_1 = df.attr_1.astype('category')

    

    df.loc[df.attr_2.isna(), 'attr_2'] = -1

    df.attr_2 = df.attr_2.astype('category')

    

    df.loc[df.attr_3.isna(), 'attr_3'] = -1

    df.attr_3 = df.attr_3.astype('category')



    df.loc[df.age.isna(), 'age'] = df.age.mean()

    df.age = df.age.astype(int)

    df.age = df.age.clip(10, 78)

    

    df.country_code = df.country_code.astype('category')



    if 'open_flag' in df.columns:

        df.loc[df.open_flag == -1, 'open_flag'] = 0



    df.loc[(df.last_open_day < 10) & (df.last_open_day >= 0) & (df.open_count_last_10_days <= 0), 'open_count_last_10_days'] = 1

    df.loc[(df.last_login_day < 10) & (df.last_login_day >= 0) & (df.login_count_last_10_days <= 0), 'login_count_last_10_days'] = 1

    df.loc[(df.last_checkout_day < 10) & (df.last_checkout_day >= 0) & (df.checkout_count_last_10_days <= 0), 'checkout_count_last_10_days'] = 1



    df.loc[(df.last_open_day < 30) & (df.last_open_day >= 0) & (df.open_count_last_30_days <= 0), 'open_count_last_30_days'] = df.loc[(df.last_open_day < 30) & (df.last_open_day >= 0) & (df.open_count_last_30_days <= 0), 'open_count_last_10_days']

    df.loc[(df.last_login_day < 30) & (df.last_login_day >= 0) & (df.login_count_last_30_days <= 0), 'login_count_last_30_days'] = df.loc[(df.last_login_day < 30) & (df.last_login_day >= 0) & (df.login_count_last_30_days <= 0), 'login_count_last_10_days']

    df.loc[(df.last_checkout_day < 30) & (df.last_checkout_day >= 0) & (df.checkout_count_last_30_days <= 0), 'checkout_count_last_30_days'] = df.loc[(df.last_checkout_day < 30) & (df.last_checkout_day >= 0) & (df.checkout_count_last_30_days <= 0), 'checkout_count_last_10_days']

    

    df.loc[(df.last_open_day < 60) & (df.last_open_day >= 0) & (df.open_count_last_60_days <= 0), 'open_count_last_60_days'] = df.loc[(df.last_open_day < 60) & (df.last_open_day >= 0) & (df.open_count_last_60_days <= 0), 'open_count_last_10_days']

    df.loc[(df.last_login_day < 60) & (df.last_login_day >= 0) & (df.login_count_last_60_days <= 0), 'login_count_last_60_days'] = df.loc[(df.last_login_day < 60) & (df.last_login_day >= 0) & (df.login_count_last_60_days <= 0), 'login_count_last_10_days']

    df.loc[(df.last_checkout_day < 60) & (df.last_checkout_day >= 0) & (df.checkout_count_last_60_days <= 0), 'checkout_count_last_60_days'] = df.loc[(df.last_checkout_day < 60) & (df.last_checkout_day >= 0) & (df.checkout_count_last_60_days <= 0), 'checkout_count_last_10_days']

    

    return df



def numerical_to_categorical(df, df_out):

    cols = [

    'subject_line_length', 'last_open_day', 'last_login_day', 'last_checkout_day',

    'open_count_last_10_days', 'open_count_last_30_days', 'open_count_last_60_days', 

    'login_count_last_10_days', 'login_count_last_30_days', 'login_count_last_60_days', 

    'checkout_count_last_10_days', 'checkout_count_last_30_days', 'checkout_count_last_60_days', 'age'

    ]



    for i, col in enumerate(cols):

        bins = [0, df[col].quantile(0.1), df[col].quantile(0.2), df[col].quantile(0.3), df[col].quantile(0.4), df[col].quantile(0.5), df[col].quantile(0.6), df[col].quantile(0.7), df[col].quantile(0.8), df[col].quantile(0.9)]

        names = [

            "<{}".format(df[col].quantile(0.1)), 

            "{}-{}".format(df[col].quantile(0.1), df[col].quantile(0.2)),

            "{}-{}".format(df[col].quantile(0.2), df[col].quantile(0.3)),

            "{}-{}".format(df[col].quantile(0.3), df[col].quantile(0.4)), 

            "{}-{}".format(df[col].quantile(0.4), df[col].quantile(0.5)),

            "{}-{}".format(df[col].quantile(0.5), df[col].quantile(0.6)),

            "{}-{}".format(df[col].quantile(0.6), df[col].quantile(0.7)),

            "{}-{}".format(df[col].quantile(0.8), df[col].quantile(0.9)),

            ">{}".format(df[col].quantile(0.9))]

        d = dict(enumerate(names, 1))

        df_out[col] = np.vectorize(d.get)(np.digitize(df_out[col], bins))

        df_out[col] = df_out[col].astype('category')

    

    return df_out
df = df.sample(frac=1)

df = clean_data(df)

# df = numerical_to_categorical(df, df)

df
def encode_train_data(df, num_cols, cat_cols):

    encoder = OneHotEncoder(handle_unknown='ignore',sparse=False, categories='auto')

    categorical_data = df.loc[:,cat_cols].values

    categorical_data = encoder.fit_transform(categorical_data).astype(np.float64)

    numerical_data = df.loc[:,num_cols].values.astype(np.float64)

    return np.concatenate([categorical_data,numerical_data],axis=1), encoder

    

def encode_test_data(df, encoder, num_cols, cat_cols):

    categorical_data = df.loc[:,cat_cols].values

    categorical_data = encoder.transform(categorical_data).astype(np.float64)

    numerical_data = df.loc[:,num_cols].values.astype(np.float64)

    return np.concatenate([categorical_data,numerical_data],axis=1)



def binarize(y_pred):

    for i in range(len(y_pred)):

        if y_pred[i]>=.5:

            y_pred[i]=1

        else:  

            y_pred[i]=0



    return np.rint(y_pred)



def get_labels(cat_cols, num_cols):

    i = 0

    labels = {}



    for cat in cat_cols:

        for val in sorted(df[cat].unique()):

            labels["f{}".format(i)] = "{}_{}".format(cat, val)

            i += 1

    

    for cat in num_cols:

        labels["f{}".format(i)] = cat

        i += 1

        

    return labels
cat_cols = [

    'country_code', 'attr_1', 'attr_2', 'attr_3', 'domain', 'day_of_week', 'is_hol', 'user_id_len', 'quarter_of_month'

]

num_cols = [

    'subject_line_length', 'last_open_day', 'last_login_day', 'last_checkout_day',

    'open_count_last_10_days', 'open_count_last_30_days', 'open_count_last_60_days', 

    'login_count_last_10_days', 'login_count_last_30_days', 'login_count_last_60_days', 

    'checkout_count_last_10_days', 'checkout_count_last_30_days', 'checkout_count_last_60_days', 'age'

    ]
params = {

    'subsample': 1.0, 

    'scale_pos_weight': 3, 

    'reg_lambda': 5.0, 

    'n_estimators': 900, 

    'min_child_weight': 5.0, 

    'max_depth': 5, 

    'gamma': 0, 

    'colsample_bytree': 0.8,

    'tree_method': 'auto', 

    'learning_rate': 0.05, 

    'objective': 'binary:logistic', 

    'verbosity': 0,             

    'nthread': -1, 

    'random_state': 42, 

    'eval_metric': "auc",

    'booster': 'dart',

}
X = df[cat_cols+num_cols]

y = df['open_flag']



skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state = 42)

xgb_model = xgb.XGBClassifier(**params)



for train_idx, test_idx in skf.split(X, y):

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]

    

    X_train, encoder = encode_train_data(X_train, num_cols, cat_cols)

    X_test = encode_test_data(X_test, encoder, num_cols, cat_cols)

    y_train, y_test = y[train_idx], y[test_idx]

    

    xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)])
labels = get_labels(cat_cols, num_cols)



fig, ax = plt.subplots(1,1,figsize=(10,25))

plt.barh(list(labels.values()), xgb_model.feature_importances_)

plt.show()
X = df[cat_cols+num_cols]

y = df['open_flag']



thresholds = sort(xgb_model.feature_importances_)



print('{} thresholds found.'.format(len(thresholds)))
results = []



for thresh in thresholds[:16]:

    selection = SelectFromModel(xgb_model, threshold=thresh, prefit=True)

    skf = StratifiedKFold(n_splits=10, random_state = 42)

    xgb_model_2 = xgb.XGBClassifier()

    xgb_model_2.set_params(**params)



    for train_idx, test_idx in skf.split(X, y):

        X_train, X_test = X.loc[train_idx], X.loc[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]



        X_train, encoder = encode_train_data(X_train, num_cols, cat_cols)

        X_train = selection.transform(X_train)



        X_test = encode_test_data(X_test, encoder, num_cols, cat_cols)

        X_test = selection.transform(X_test)



        xgb_model_2.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)])

        

    y_pred = xgb_model_2.predict(X_test)

    y_pred = binarize(y_pred)

    accuracy = matthews_corrcoef(y_pred,y_test)



    results.append((thresh, X_train.shape[1], accuracy*100.0))
for result in results:

    print("Thresh=%.6f, n=%d, Accuracy: %.2f%%" % result)
df_val = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/test.csv')

users = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/users.csv')

df_val = pd.merge(df_val, users, on='user_id')

df = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/train.csv', index_col='row_id')

df = pd.merge(df, users, on='user_id')

df = clean_data(df)



df_val = clean_data(df_val)

# df_val = numerical_to_categorical(df, df_val)
X_val = encode_test_data(df_val, encoder, num_cols, cat_cols)

y_preds = xgb_model.predict(X_val)

y_preds = binarize(y_preds)

df_val['open_flag'] = y_preds

df_val = df_val[['row_id', 'open_flag']]

df_val.open_flag = df_val.open_flag.astype(int)

df_val
df_val.open_flag.value_counts()
df_val.to_csv('submission.csv', index=False)