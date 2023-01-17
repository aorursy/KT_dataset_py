import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/cat-in-the-dat/train.csv', index_col='id')
train.shape
train.head().T
test = pd.read_csv('../input/cat-in-the-dat/test.csv', index_col='id')
test.shape
fa_features = [

    'bin_0', 'bin_1',

    'nom_5', 'nom_6'

]



train_fa = train[fa_features].copy()

test_fa = test[fa_features].copy()
def summary(df):

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name', 'dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    return summary





summary(train)
ohe_features = [

    'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',

    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',

    'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4',

    'day', 'month'

]



le_features = list(set(test.columns) - set(ohe_features))
train_part = len(train)

df = pd.get_dummies(pd.concat([train, test], axis=0), columns=ohe_features)

train = df[:train_part]

test = df[train_part:].drop('target', axis=1)

del df
from sklearn.preprocessing import LabelEncoder





def encode_categorial_features_fit(df, columns_to_encode):

    encoders = {}

    for c in columns_to_encode:

        if c in df.columns:

            encoder = LabelEncoder()

            encoder.fit(df[c].astype(str).values)

            encoders[c] = encoder

    return encoders



def encode_categorial_features_transform(df, encoders):

    out = pd.DataFrame(index=df.index)

    for c in encoders.keys():

        if c in df.columns:

            out[c] = encoders[c].transform(df[c].astype(str).values)

    return out





# le_features = test.columns



categorial_features_encoders = encode_categorial_features_fit(

    pd.concat([train, test], join='outer', sort=False), le_features)
temp = encode_categorial_features_transform(train, categorial_features_encoders)

columns_to_drop = list(set(le_features) & set(train.columns))

train = train.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)

del temp
temp = encode_categorial_features_transform(test, categorial_features_encoders)

columns_to_drop = list(set(le_features) & set(test.columns))

test = test.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)

del temp
from category_encoders import TargetEncoder
te_features = [

    'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

]
te = TargetEncoder(cols=te_features, drop_invariant=True, return_df=True, min_samples_leaf=2, smoothing=1.0)

te.fit(train[te_features], train['target'])
temp = te.transform(train[te_features])

columns_to_drop = list(set(te_features) & set(train.columns))

train = train.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)

del temp
temp = te.transform(test[te_features])

columns_to_drop = list(set(te_features) & set(test.columns))

test = test.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)

del temp
le_features = fa_features



categorial_features_encoders = encode_categorial_features_fit(

    pd.concat([train_fa, test_fa], join='outer', sort=False), le_features)
temp = encode_categorial_features_transform(train_fa, categorial_features_encoders)

columns_to_drop = list(set(le_features) & set(train_fa.columns))

train_fa = train_fa.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)

del temp
temp = encode_categorial_features_transform(test_fa, categorial_features_encoders)

columns_to_drop = list(set(le_features) & set(test_fa.columns))

test_fa = test_fa.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)

del temp
def make_aggregates(df, feature_to_group_by, feature):

    out = pd.DataFrame(index=df.index)

    agg = df.groupby([feature_to_group_by])[feature].value_counts(normalize=True)

    freq = lambda row: agg.loc[row[feature_to_group_by], row[feature]]

    out[feature + '__' + feature_to_group_by + '_freq'] = df.apply(freq, axis=1)

    return out





for feature in ['nom_5__bin_0', 'nom_6__bin_1']:

    feature_1, feature_2 = feature.split('__')

    print('Add feature:', feature, '/ aggregates of', feature_2, 'by', feature_1)

    

    agg = make_aggregates(train_fa, feature_2, feature_1)

    train = train.merge(agg, how='left', left_index=True, right_index=True)

    del agg

    

    agg = make_aggregates(test_fa, feature_2, feature_1)

    test = test.merge(agg, how='left', left_index=True, right_index=True)

    del agg



del train_fa

del test_fa
# From https://www.kaggle.com/gemartin/load-data-reduce-memory-usage



def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

#        else:

#            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df





train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
y_train = train['target'].copy()

x_train = train.drop('target', axis=1)

del train



x_test = test.copy()

del test
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from bayes_opt import BayesianOptimization
def train_model(num_leaves, min_data_in_leaf, max_depth, bagging_fraction, feature_fraction, lambda_l1, lambda_l2):

    

    params = {

        'objective': 'binary',

        'metric': 'auc',

        'boosting_type': 'gbdt',

        'is_unbalance': False,

        'boost_from_average': True,

        'num_threads': 4,

        

        'num_leaves': int(num_leaves),

        'min_data_in_leaf': int(min_data_in_leaf),

        'max_depth': int(max_depth),

        'bagging_fraction' : bagging_fraction,

        'feature_fraction' : feature_fraction,

        'lambda_l1': lambda_l1,

        'lambda_l2': lambda_l2

    }

    

    scores = []

    

    cv = KFold(n_splits=10, random_state=42)

    for train_idx, valid_idx in cv.split(x_train, y_train):

        

        x_train_train = x_train.iloc[train_idx]

        y_train_train = y_train.iloc[train_idx]

        x_train_valid = x_train.iloc[valid_idx]

        y_train_valid = y_train.iloc[valid_idx]

        

        lgb_train = lgb.Dataset(data=x_train_train.astype('float32'), label=y_train_train.astype('float32'))

        lgb_valid = lgb.Dataset(data=x_train_valid.astype('float32'), label=y_train_valid.astype('float32'))

        

        lgb_model = lgb.train(params, lgb_train, valid_sets=lgb_valid, verbose_eval=100)

        y = lgb_model.predict(x_train_valid.astype('float32'), num_iteration=lgb_model.best_iteration)

        

        score = roc_auc_score(y_train_valid.astype('float32'), y)

        print('Fold score:', score)

        scores.append(score)

    

    average_score = sum(scores) / len(scores)

    print('Average score:', average_score)

    return average_score





bounds = {

    'num_leaves': (31, 100),

    'min_data_in_leaf': (20, 100),

    'max_depth':(-1, 100),

    'bagging_fraction' : (0.1, 0.9),

    'feature_fraction' : (0.1, 0.9),

    'lambda_l1': (0, 2),

    'lambda_l2': (0, 2)

}



bo = BayesianOptimization(train_model, bounds, random_state=42)

bo.maximize(init_points=20, n_iter=20, acq='ucb', xi=0.0, alpha=1e-6)
bo.max
params = {

    'objective': 'binary',

    'metric': 'auc',

    'boosting_type': 'gbdt',

    'is_unbalance': False,

    'boost_from_average': True,

    'num_threads': 4,

    

    'num_iterations': 10000,

    'learning_rate': 0.006,

    'early_stopping_round': 100,

    

    'num_leaves': int(bo.max['params']['num_leaves']),

    'min_data_in_leaf': int(bo.max['params']['min_data_in_leaf']),

    'max_depth': int(bo.max['params']['max_depth']),

    'bagging_fraction' : bo.max['params']['bagging_fraction'],

    'feature_fraction' : bo.max['params']['feature_fraction'],

    'lambda_l1': bo.max['params']['lambda_l1'],

    'lambda_l2': bo.max['params']['lambda_l2']

    

#    'num_leaves': 94,

#    'min_data_in_leaf': 61,

#    'max_depth': 31,

#    'bagging_fraction' : 0.12033530139527615,

#    'feature_fraction' : 0.18631314159464357,

#    'lambda_l1': 0.0628583713734685,

#    'lambda_l2': 1.2728208225275608

}
n_splits = 10



y = np.zeros(x_test.shape[0])

oof = np.zeros(x_train.shape[0])

feature_importances = []



cv = KFold(n_splits=n_splits, random_state=42)

for train_idx, valid_idx in cv.split(x_train, y_train):

    

    x_train_train = x_train.iloc[train_idx]

    y_train_train = y_train.iloc[train_idx]

    x_train_valid = x_train.iloc[valid_idx]

    y_train_valid = y_train.iloc[valid_idx]

    

    lgb_train = lgb.Dataset(data=x_train_train.astype('float32'), label=y_train_train.astype('float32'))

    lgb_valid = lgb.Dataset(data=x_train_valid.astype('float32'), label=y_train_valid.astype('float32'))

    

    lgb_model = lgb.train(params, lgb_train, valid_sets=lgb_valid, verbose_eval=100)

    

    y_part = lgb_model.predict(x_test.astype('float32'), num_iteration=lgb_model.best_iteration)

    y += y_part / n_splits

    

    oof_part = lgb_model.predict(x_train_valid.astype('float32'), num_iteration=lgb_model.best_iteration)

    oof[valid_idx] = oof_part

    

    score = roc_auc_score(y_train_valid.astype('float32'), oof_part)

    print('Fold score:', score)

    

    feature_importances.append(lgb_model.feature_importance())
import matplotlib.pyplot as plt

import seaborn as sns



feature_importance_df = pd.concat([

    pd.Series(x_train.columns),

    pd.Series(np.mean(feature_importances, axis=0))], axis=1)

feature_importance_df.columns = ['featureName', 'importance']



temp = feature_importance_df.sort_values(by=['importance'], ascending=False)



plt.figure(figsize=(12, 20))

sns.barplot(x="importance", y="featureName", data=temp)

plt.show()
temp = feature_importance_df.sort_values(by=['importance'], ascending=False).head(15)

most_important_features = temp['featureName'].values
plt.figure(figsize=(20,20))

cor = x_train[most_important_features].corr()

sns.heatmap(cor, annot=True, annot_kws={"size": 8}, cmap=plt.cm.Reds)

plt.show()
# From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



import matplotlib.pyplot as plt



from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels





def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
classes = pd.Series([0,1])



plot_confusion_matrix(y_train, oof.round(), classes=classes, normalize=True, title='Confusion matrix')



plt.show()
submission = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv', index_col='id')

submission['target'] = y

submission.to_csv('lightgbm.csv')
submission.head()
oof_df = pd.DataFrame(index=pd.read_csv('../input/cat-in-the-dat/train.csv', index_col='id').index)

oof_df['oof'] = oof

oof_df.to_csv('oof.csv')
oof_df.head()