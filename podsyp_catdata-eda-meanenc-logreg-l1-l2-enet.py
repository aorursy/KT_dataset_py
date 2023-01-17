import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")



import gc



from scipy.stats import ttest_ind, ttest_rel

from scipy import stats
from sklearn.base import BaseEstimator

from sklearn.impute import SimpleImputer as Imputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('use_inf_as_na', True)



warnings.simplefilter('ignore')

matplotlib.rcParams['figure.dpi'] = 300

sns.set()

%matplotlib inline
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

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
%%time

folder = '../input/cat-in-the-dat-ii/'

train_df = reduce_mem_usage(pd.read_csv(folder + 'train.csv'))

test_df = reduce_mem_usage(pd.read_csv(folder + 'test.csv'))

sub_df = reduce_mem_usage(pd.read_csv(folder + 'sample_submission.csv'))
print('train')

print('All: ', train_df.shape)

print('test')

print('All: ', test_df.shape)

print('sub')

print('sub_df ', sub_df.shape)
train_df.head()
test_df.head()
# Target
y = train_df['target']

train_df.drop(['target'], axis=1, inplace=True)
y.value_counts()
y.value_counts(normalize=True)
y.hist(bins=len(y.value_counts()));
train_df['day_month'] = train_df['month'] * 100 + train_df['day']

test_df['day_month'] = test_df['month'] * 100 + train_df['day']
train_df['bin_3'] = train_df['bin_3'].apply(lambda x: 1 if x == 'T' else 0)

train_df['bin_4'] = train_df['bin_4'].apply(lambda x: 1 if x == 'Y' else 0)

test_df['bin_3'] = test_df['bin_3'].apply(lambda x: 1 if x == 'T' else 0)

test_df['bin_4'] = test_df['bin_4'].apply(lambda x: 1 if x == 'Y' else 0)
train_df.drop(['id'], axis=1, inplace=True)

test_df.drop(['id'], axis=1, inplace=True)
train_df.describe()
test_df.describe()
num_cols = test_df.describe().columns.tolist()
train_df.describe(include=['O'])
test_df.describe(include=['O'])
cat_cols = test_df.describe(include=['O']).columns.tolist()
def missing_values_table(df, info=True):

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        if info:

            print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

                "There are " + str(mis_val_table_ren_columns.shape[0]) +

                  " columns that have missing values.")

        return mis_val_table_ren_columns.T
print('Numeric columns: ')

missing_values_table(train_df[num_cols])
print('Numeric columns: ')

missing_values_table(test_df[num_cols])
print('Categorical columns: ')

missing_values_table(train_df[cat_cols])
print('Categorical columns: ')

missing_values_table(test_df[cat_cols])
class MeanEncoding(BaseEstimator):

    """   In Mean Encoding we take the number 

    of labels into account along with the target variable 

    to encode the labels into machine comprehensible values    """

    

    def __init__(self, feature, C=0.1):

        self.C = C

        self.feature = feature

        

    def fit(self, X_train, y_train):

        

        df = pd.DataFrame({'feature': X_train[self.feature], 'target': y_train}).dropna()

        

        self.global_mean = df.target.mean()

        mean = df.groupby('feature').target.mean()

        size = df.groupby('feature').target.size()

        

        self.encoding = (self.global_mean * self.C + mean * size) / (self.C + size)

    

    def transform(self, X_test):

        

        X_test[self.feature] = X_test[self.feature].map(self.encoding).fillna(self.global_mean).values

        

        return X_test

    

    def fit_transform(self, X_train, y_train):

        

        df = pd.DataFrame({'feature': X_train[self.feature], 'target': y_train}).dropna()

        

        self.global_mean = df.target.mean()

        mean = df.groupby('feature').target.mean()

        size = df.groupby('feature').target.size()

        self.encoding = (self.global_mean * self.C + mean * size) / (self.C + size)

        

        X_train[self.feature] = X_train[self.feature].map(self.encoding).fillna(self.global_mean).values

        

        return X_train
for f in cat_cols+['day', 'month']:

    me = MeanEncoding(f, C=0.01*len(train_df[f].unique()))

    me.fit(train_df, y)

    train_df = me.transform(train_df)

    test_df = me.transform(test_df)
imputer = Imputer(strategy="mean")

imputer.fit(train_df)

train_df = pd.DataFrame(imputer.transform(train_df), columns=train_df.columns)

test_df = pd.DataFrame(imputer.transform(test_df), columns=train_df.columns)
train_df['bin_sum'] = train_df['bin_0'] + train_df['bin_1'] + train_df['bin_2'] + train_df['bin_3'] + train_df['bin_4']

test_df['bin_sum'] = test_df['bin_0'] + test_df['bin_1'] + test_df['bin_2'] + test_df['bin_3'] + test_df['bin_4']
train_df['nom_sum'] = train_df['nom_0'] + train_df['nom_1'] + train_df['nom_2'] + train_df['nom_3'] + train_df['nom_4'] + train_df['nom_5'] + train_df['nom_6'] + train_df['nom_7'] + train_df['nom_8'] + train_df['nom_9']

test_df['nom_sum'] = test_df['nom_0'] + test_df['nom_1'] + test_df['nom_2'] + test_df['nom_3'] + test_df['nom_4'] + test_df['nom_5'] + test_df['nom_6'] + test_df['nom_7'] + test_df['nom_8'] + test_df['nom_9']
train_df['nom_multi'] = train_df['nom_0'] * train_df['nom_1'] * train_df['nom_2'] * train_df['nom_3'] * train_df['nom_4'] * train_df['nom_5'] * train_df['nom_6'] * train_df['nom_7'] * train_df['nom_8'] * train_df['nom_9']

test_df['nom_multi'] = test_df['nom_0'] * test_df['nom_1'] * test_df['nom_2'] * test_df['nom_3'] * test_df['nom_4'] * test_df['nom_5'] * test_df['nom_6'] * test_df['nom_7'] * test_df['nom_8'] * test_df['nom_9']
train_df['ord_sum'] = train_df['ord_0'] + train_df['ord_1'] + train_df['ord_2'] + train_df['ord_3'] + train_df['ord_4'] + train_df['ord_5']

test_df['ord_sum'] = test_df['ord_0'] + test_df['ord_1'] + test_df['ord_2'] + test_df['ord_3'] + test_df['ord_4'] + test_df['ord_5']
train_df['ord_multi'] = train_df['ord_0'] * train_df['ord_1'] * train_df['ord_2'] * train_df['ord_3'] * train_df['ord_4'] * train_df['ord_5']

test_df['ord_multi'] = test_df['ord_0'] * test_df['ord_1'] * test_df['ord_2'] * test_df['ord_3'] * test_df['ord_4'] * test_df['ord_5']
train_corr = train_df.corr()

# plot the heatmap and annotation on it

fig, ax = plt.subplots(figsize=(14,14))

sns.heatmap(train_corr, xticklabels=train_corr.columns, yticklabels=train_corr.columns, annot=True, ax=ax);
test_corr = test_df.corr()

# plot the heatmap and annotation on it

fig, ax = plt.subplots(figsize=(14,14))

sns.heatmap(test_corr, xticklabels=test_corr.columns, yticklabels=test_corr.columns, annot=True, ax=ax);
train_df.drop(['ord_0', 'month', 'nom_sum', ], axis=1, inplace=True)

test_df.drop(['ord_0', 'month', 'nom_sum', ], axis=1, inplace=True)
%%time



skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=777)



features_stats = []



for c in train_df.columns:

    ks = stats.ks_2samp(train_df[c], test_df[c])

    mv = stats.mannwhitneyu(train_df[c], test_df[c])

    

    train_score = []

    test_score = []

    

    for train_index, val_index in skf.split(train_df, y):

        x_train, x_valid = train_df.iloc[train_index, :][[c]], train_df.iloc[val_index, :][[c]]

        y_train, y_valid = y[train_index], y[val_index]

        

        logreg = LogisticRegression()

        logreg.fit(x_train, y_train)

        train_score.append(roc_auc_score(y_train, logreg.predict_proba(x_train)[:, 1]))

        test_score.append(roc_auc_score(y_valid, logreg.predict_proba(x_valid)[:, 1]))

        

    train_score_ = np.mean(train_score)

    test_score_ = np.mean(test_score)

    

    features_stats.append([c, train_score_, test_score_, ks[0], ks[1], mv[0], mv[1]])

    

features_stats = pd.DataFrame(features_stats, columns=['Name', 'Train_AUC', 'Test_AUC', 'KS_Stats', 'KS_pvalue', 'MV_Stats', 'MV_pvalue'])
features_stats.sort_values('Test_AUC', ascending=False)
scaler = StandardScaler()

train_df_scale = scaler.fit_transform(train_df)

test_df_scale = scaler.transform(test_df)
%%time

train_scores=[]

test_scores=[]



skf_1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)



lr_grid_l1 = {"C":[100, 10, 1, 0.5, 0.1, 0.5, 0.25, 0.05, 0.01, 0.005, 0.001], "penalty":['l1'], "solver":['liblinear',]}

lr_grid_l2 = {"C":[100, 10, 1, 0.5, 0.1, 0.5, 0.25, 0.05, 0.01, 0.005, 0.001], "penalty":['l2'], "solver":['newton-cg','lbfgs', 'saga']}

lr_grid_el = {"C":[1, 0.5, 0.1, 0.5, 0.25, 0.05, 0.01, 0.005, 0.001], 

              "l1_ratio":[1, 0.5, 0.1, 0.5, 0.25, 0.05, 0.01, 0.005, 0.001], "penalty":['elasticnet'], "solver":['saga']}



for no, (train_index_1, val_index_1) in enumerate(skf_1.split(train_df_scale, y)):

    x_train, x_valid = train_df_scale[train_index_1, :], train_df_scale[val_index_1, :]

    y_train, y_valid = y[train_index_1], y[val_index_1]

    

    logreg_l1=LogisticRegression()

    logreg_cv_l1=RandomizedSearchCV(logreg_l1, lr_grid_l1, cv=5, verbose=False, scoring='roc_auc', n_jobs=-1)

    logreg_cv_l1.fit(x_train, y_train)

    logreg_model_l1 = LogisticRegression(**logreg_cv_l1.best_params_).fit(x_train, y_train)

    train_pred_l1 = logreg_model_l1.predict_proba(train_df_scale)[:, 1]

    test_pred_l1 = logreg_model_l1.predict_proba(test_df_scale)[:, 1]

    train_scores.append(train_pred_l1)

    test_scores.append(test_pred_l1)

    

    print('Fold Log L1: ', no, 'CV AUC: ', logreg_cv_l1.best_score_, 

          'Best params: ', logreg_cv_l1.best_params_)

    

    logreg_l2=LogisticRegression()

    logreg_cv_l2=RandomizedSearchCV(logreg_l2, lr_grid_l2, cv=3, verbose=False, scoring='roc_auc', n_jobs=-1)

    logreg_cv_l2.fit(x_train, y_train)

    logreg_model_l2 = LogisticRegression(**logreg_cv_l2.best_params_).fit(x_train, y_train)

    train_pred_l2 = logreg_model_l2.predict_proba(train_df_scale)[:, 1]

    test_pred_l2 = logreg_model_l2.predict_proba(test_df_scale)[:, 1]

    train_scores.append(train_pred_l2)

    test_scores.append(test_pred_l2)

    print('Fold Log L2: ', no, 'CV AUC: ', logreg_cv_l2.best_score_, 

          'Best params: ', logreg_cv_l2.best_params_)

        

    logreg_el=LogisticRegression()

    logreg_cv_el=RandomizedSearchCV(logreg_el, lr_grid_el, cv=3, verbose=False, scoring='roc_auc', n_jobs=-1)

    logreg_cv_el.fit(x_train, y_train)

    logreg_model_el = LogisticRegression(**logreg_cv_el.best_params_).fit(x_train, y_train)

    train_pred_el = logreg_model_el.predict_proba(train_df_scale)[:, 1]

    test_pred_el = logreg_model_el.predict_proba(test_df_scale)[:, 1]

    train_scores.append(train_pred_el)

    test_scores.append(test_pred_el)

    print('Fold Log EL: ', no, 'CV AUC: ', logreg_cv_el.best_score_, 

          'Best params: ', logreg_cv_el.best_params_)
logreg=LogisticRegression(C=0.01)

logreg.fit(np.array(train_scores).T, y)

sub_df['target'] = logreg.predict_proba(np.array(test_scores).T)[:, 1]
sub_df.to_csv('submission.csv', index=False)