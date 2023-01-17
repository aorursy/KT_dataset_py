import numpy as np 

import pandas as pd

import os

import xgboost

import gc

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier 



import lightgbm as lgb

from numba import jit 



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')

test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')

train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
train_df.head(10)
train_labels_df.head(10)
specs_df.head(10)
def extract_time_features(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['date'] = df['timestamp'].dt.date

    df['month'] = df['timestamp'].dt.month

    df['hour'] = df['timestamp'].dt.hour

    df['year'] = df['timestamp'].dt.year

    df['dayofweek'] = df['timestamp'].dt.dayofweek

    df['weekofyear'] = df['timestamp'].dt.weekofyear

    df['dayofyear'] = df['timestamp'].dt.dayofyear

    df['quarter'] = df['timestamp'].dt.quarter

    df['is_month_start'] = df['timestamp'].dt.is_month_start    

    return df
def get_object_columns(df, columns):

    df = df.groupby(['installation_id', columns])['event_id'].count().reset_index()

    df = df.pivot_table(index = 'installation_id', columns = [columns], values = 'event_id')

    df.columns = list(df.columns)

    df.fillna(0, inplace = True)

    return df



def get_numeric_columns(df, column):

    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']})

    df[column].fillna(df[column].mean(), inplace = True)

    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_min', f'{column}_max', f'{column}_std', f'{column}_skew']

    return df



def get_numeric_columns_add(df, agg_column, column):

    df = df.groupby(['installation_id', agg_column]).agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']}).reset_index()

    df = df.pivot_table(index = 'installation_id', columns = [agg_column], values = [col for col in df.columns if col not in ['installation_id', 'type']])

    df[column].fillna(df[column].mean(), inplace = True)

    df.columns = list(df.columns)

    return df
def perform_features_engineering(train_df, test_df, train_labels_df):

    print(f'Perform features engineering')

    numerical_columns = ['game_time']

    categorical_columns = ['type', 'world']



    comp_train_df = pd.DataFrame({'installation_id': train_df['installation_id'].unique()})

    comp_train_df.set_index('installation_id', inplace = True)

    comp_test_df = pd.DataFrame({'installation_id': test_df['installation_id'].unique()})

    comp_test_df.set_index('installation_id', inplace = True)



    test_df = extract_time_features(test_df)

    train_df = extract_time_features(train_df)



    for i in numerical_columns:

        comp_train_df = comp_train_df.merge(get_numeric_columns(train_df, i), left_index = True, right_index = True)

        comp_test_df = comp_test_df.merge(get_numeric_columns(test_df, i), left_index = True, right_index = True)

    

    for i in categorical_columns:

        comp_train_df = comp_train_df.merge(get_object_columns(train_df, i), left_index = True, right_index = True)

        comp_test_df = comp_test_df.merge(get_object_columns(test_df, i), left_index = True, right_index = True)

    

    for i in categorical_columns:

        for j in numerical_columns:

            comp_train_df = comp_train_df.merge(get_numeric_columns_add(train_df, i, j), left_index = True, right_index = True)

            comp_test_df = comp_test_df.merge(get_numeric_columns_add(test_df, i, j), left_index = True, right_index = True)

    

    

    comp_train_df.reset_index(inplace = True)

    comp_test_df.reset_index(inplace = True)

       

    labels_map = dict(train_labels_df.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))

 

    labels = train_labels_df[['installation_id', 'title', 'accuracy_group']]

    

    labels['title'] = labels['title'].map(labels_map)

   

    comp_test_df['title'] = test_df.groupby('installation_id').last()['title'].map(labels_map).reset_index(drop = True)

   

    comp_train_df = labels.merge(comp_train_df, on = 'installation_id', how = 'left')

    print('We have {} training rows'.format(comp_train_df.shape[0]))

    

    return comp_train_df, comp_test_df
def qwk3(a1, a2, max_rat=3):

    assert(len(a1) == len(a2))

    a1 = np.asarray(a1, dtype=int)

    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))

    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)

    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e
ada_train_df, ada_test_df = perform_features_engineering(train_df, test_df, train_labels_df)
null_columns = ada_test_df.columns[ada_test_df.isnull().any()]

ada_test_df[null_columns].isnull().sum()
ada_test_df['game_time_std'] = ada_test_df['game_time_std'].fillna(0)

ada_test_df['game_time_skew'] = ada_test_df['game_time_skew'].fillna(0)
def adaboost_it(ada_train_df, ada_test_df):

    print("Ada-Boosting...")

    t_splits = 5

    k_scores = []

    kf = KFold(n_splits = t_splits)

    features = [i for i in ada_train_df.columns if i not in ['accuracy_group', 'installation_id']]

    target = 'accuracy_group'

    oof_pred = np.zeros((len(ada_train_df), 4))

    y_pred = np.zeros((len(ada_test_df), 4))

    for fold, (tr_ind, val_ind) in enumerate(kf.split(ada_train_df)):

        print(f'Fold: {fold+1}')

        x_train, x_val = ada_train_df[features].iloc[tr_ind], ada_train_df[features].iloc[val_ind]

        y_train, y_val = ada_train_df[target][tr_ind], ada_train_df[target][val_ind]

               

        ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,algorithm="SAMME.R", learning_rate=0.5)

        ada_clf.fit(x_train, y_train)

        oof_pred[val_ind] = ada_clf.predict_proba(x_val)

      

        y_pred += ada_clf.predict_proba(ada_test_df[features]) / t_splits

        

        val_crt_fold = qwk3(y_val, oof_pred[val_ind].argmax(axis = 1))

        print(f'Fold: {fold+1} quadratic weighted kappa score: {np.round(val_crt_fold,4)}')

        

    res = qwk3(ada_train_df['accuracy_group'], oof_pred.argmax(axis = 1))

    print(f'Quadratic weighted score: {np.round(res,4)}')

        

    return y_pred
y_pred = adaboost_it(ada_train_df, ada_test_df)
ada_test_df = ada_test_df.reset_index()

ada_test_df = ada_test_df[['installation_id']]

ada_test_df['accuracy_group'] = y_pred.argmax(axis = 1)

ada_sample_submission_df = sample_submission_df.merge(ada_test_df, on = 'installation_id')

ada_sample_submission_df.to_csv('ada_boost_submission.csv', index = False)
xgb_train_df, xgb_test_df = perform_features_engineering(train_df, test_df, train_labels_df)
features = [i for i in xgb_train_df.columns if i not in ['accuracy_group', 'installation_id']]

target = 'accuracy_group'
x_train  = xgb_train_df[features]

y_train = xgb_train_df[target]
#from sklearn.model_selection import GridSearchCV

#model = xgboost.XGBClassifier()



#param_dist = {"max_depth": [10,30,50],"min_child_weight" : [1,3,6],

 #             "n_estimators": [200],

  #            "learning_rate": [0.05, 0.1,0.16],}



#grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, verbose=10, n_jobs=-1)

#grid_search.fit(x_train, y_train)

#grid_search.best_estimator_
def xgb(xgb_train_df, xgb_test_df):

    print("XG-Boosting...")

    t_splits = 5

    k_scores = []

    kf = KFold(n_splits = t_splits)

    features = [i for i in xgb_train_df.columns if i not in ['accuracy_group', 'installation_id']]

    target = 'accuracy_group'

    oof_pred = np.zeros((len(xgb_train_df), 4))

    y_pred = np.zeros((len(xgb_test_df), 4))

    for fold, (tr_ind, val_ind) in enumerate(kf.split(xgb_train_df)):

        print(f'Fold: {fold+1}')

        x_train, x_val = xgb_train_df[features].iloc[tr_ind], xgb_train_df[features].iloc[val_ind]

        y_train, y_val = xgb_train_df[target][tr_ind], xgb_train_df[target][val_ind]

        

        xgb_clf = xgboost.XGBClassifier()

        xgb_clf.fit(x_train, y_train)

        oof_pred[val_ind] = xgb_clf.predict_proba(x_val)

      

        y_pred += xgb_clf.predict_proba(xgb_test_df[features]) / t_splits

        

        val_crt_fold = qwk3(y_val, oof_pred[val_ind].argmax(axis = 1))

        print(f'Fold: {fold+1} quadratic weighted kappa score: {np.round(val_crt_fold,4)}')

        

    res = qwk3(xgb_train_df['accuracy_group'], oof_pred.argmax(axis = 1))

    print(f'Quadratic weighted score: {np.round(res,4)}')

        

    return y_pred
y_pred = xgb(xgb_train_df, xgb_test_df)
xgb_test_df = xgb_test_df.reset_index()

xgb_test_df = xgb_test_df[['installation_id']]

xgb_test_df['accuracy_group'] = y_pred.argmax(axis = 1)

xgb_sample_submission_df = sample_submission_df.merge(xgb_test_df, on = 'installation_id')

xgb_sample_submission_df.to_csv('xgb_submission.csv', index = False)
xgb_sample_submission_df = xgb_sample_submission_df.drop('accuracy_group_x', axis=1)

xgb_sample_submission_df.columns = ['installation_id', 'accuracy_group']
xgb_sample_submission_df.to_csv('xgb_submission.csv', index = False)
cat_train_df, cat_test_df = perform_features_engineering(train_df, test_df, train_labels_df)
xc_train  = cat_train_df[features]

yc_train = cat_train_df[target]
import catboost as cb

def cat(cat_train_df, cat_test_df):

    print("Meeowwww...")

    t_splits = 3

    k_scores = []

    kf = KFold(n_splits = t_splits)

    features = [i for i in cat_train_df.columns if i not in ['accuracy_group', 'installation_id']]

    target = 'accuracy_group'

    oof_pred = np.zeros((len(cat_train_df), 4))

    y_pred = np.zeros((len(cat_test_df), 4))

    for fold, (tr_ind, val_ind) in enumerate(kf.split(cat_train_df)):

        print(f'Fold: {fold+1}')

        x_train, x_val = cat_train_df[features].iloc[tr_ind], cat_train_df[features].iloc[val_ind]

        y_train, y_val = cat_train_df[target][tr_ind], cat_train_df[target][val_ind]

        

        cat_clf = cb.CatBoostClassifier(depth=10, iterations= 200, l2_leaf_reg= 9, learning_rate= 0.15)

        cat_clf.fit(xc_train, yc_train)

        oof_pred[val_ind] = cat_clf.predict_proba(x_val)

      

        y_pred += cat_clf.predict_proba(cat_test_df[features]) / t_splits

        

        val_crt_fold = qwk3(y_val, oof_pred[val_ind].argmax(axis = 1))

        print(f'Fold: {fold+1} quadratic weighted kappa score: {np.round(val_crt_fold,4)}')

        

    res = qwk3(cat_train_df['accuracy_group'], oof_pred.argmax(axis = 1))

    print(f'Quadratic weighted score: {np.round(res,4)}')

        

    return y_pred
y_pred_cat = cat(cat_train_df, cat_test_df)
cat_test_df = cat_test_df.reset_index()

cat_test_df = cat_test_df[['installation_id']]

cat_test_df['accuracy_group'] = y_pred_cat.argmax(axis = 1)

cat_sample_submission_df = sample_submission_df.merge(cat_test_df, on = 'installation_id')

cat_sample_submission_df.to_csv('submission.csv', index = False)
cat_sample_submission_df = cat_sample_submission_df.drop('accuracy_group_x', axis=1)

cat_sample_submission_df.columns = ['installation_id', 'accuracy_group']
cat_sample_submission_df.to_csv('submission.csv', index = False)
lgb_train_df, lgb_test_df = perform_features_engineering(train_df, test_df, train_labels_df)
xl_train  = lgb_train_df[features]

yl_train = lgb_train_df[target]
import lightgbm as lgb



def lgbc(lgb_train_df, lgb_test_df):

    print("Meeowwww...")

    t_splits = 3

    k_scores = []

    kf = KFold(n_splits = t_splits)

    features = [i for i in lgb_train_df.columns if i not in ['accuracy_group', 'installation_id']]

    target = 'accuracy_group'

    oof_pred = np.zeros((len(lgb_train_df), 4))

    y_pred = np.zeros((len(lgb_test_df), 4))

    for fold, (tr_ind, val_ind) in enumerate(kf.split(lgb_train_df)):

        print(f'Fold: {fold+1}')

        x_train, x_val = lgb_train_df[features].iloc[tr_ind], lgb_train_df[features].iloc[val_ind]

        y_train, y_val = lgb_train_df[target][tr_ind], lgb_train_df[target][val_ind]

        

        lg = lgb.LGBMClassifier(silent=False)

        lg.fit(xl_train, yl_train)

        oof_pred[val_ind] = lg.predict_proba(x_val)

      

        y_pred += lg.predict_proba(lgb_test_df[features]) / t_splits

        

        val_crt_fold = qwk3(y_val, oof_pred[val_ind].argmax(axis = 1))

        print(f'Fold: {fold+1} quadratic weighted kappa score: {np.round(val_crt_fold,4)}')

        

    res = qwk3(lgb_train_df['accuracy_group'], oof_pred.argmax(axis = 1))

    print(f'Quadratic weighted score: {np.round(res,4)}')

        

    return y_pred
y_pred_lgb = lgbc(lgb_train_df, lgb_test_df)
lgb_test_df = lgb_test_df.reset_index()

lgb_test_df = lgb_test_df[['installation_id']]

lgb_test_df['accuracy_group'] = y_pred_lgb.argmax(axis = 1)

lgb_sample_submission_df = sample_submission_df.merge(lgb_test_df, on = 'installation_id')

lgb_sample_submission_df.to_csv('lgb_submission.csv', index = False)
lgb_sample_submission_df = lgb_sample_submission_df.drop('accuracy_group_x', axis=1)

lgb_sample_submission_df.columns = ['installation_id', 'accuracy_group']
data = [['ada', 0.42], ['xgb', 0.44], ['cat', 0.65], ['lgb', 0.62]]



df = pd.DataFrame(data, columns = ['Model', 'Validation Kappa Score']) 
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Bar(x=df['Model'], y=df['Validation Kappa Score'], marker_color='#FFD700'))

fig.show()