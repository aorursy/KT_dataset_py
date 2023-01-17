%matplotlib inline

import pandas as pd

import numpy as np

import itertools

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.model_selection import train_test_split

import lightgbm as lgbm

from sklearn.metrics import auc, f1_score, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score

import seaborn as sns

import time

pd.set_option('display.max_rows', 90)

pd.set_option('display.max_columns', 200)

pd.set_option('display.width', 1000)
in_file = '/kaggle/input/datasets-for-churn-telecom/cell2celltrain.csv'
df = pd.read_csv(in_file)
df.head()
df.shape
df.Churn.value_counts()
df.Churn.value_counts()/df.shape[0]
# number of missing values in dataset

df.isnull().sum().values.sum()
missing = list()

for x in df.columns:

    if df[x].isnull().sum() != 0:

        print(x, df[x].isnull().sum())

        missing.append(x)
# First of, what on earthe is this feature name? is the Director assisting calls?

df.DirectorAssistedCalls.describe()
plt.figure(figsize=(7,7))

plt.grid(True)

sns.distplot(df.DirectorAssistedCalls.fillna(0))

plt.xlim(right=25)
df.Churn[df.DirectorAssistedCalls != 0].value_counts()
# ratios against all customer population

df.Churn[df.DirectorAssistedCalls != 0].value_counts()/df.shape[0]
df.AgeHH1.describe()
df.AgeHH2.describe()
df.Churn[(df.AgeHH1.fillna(0) == 0)&(df.AgeHH2.fillna(0) == 0)].value_counts()
df = df.fillna(0)
#get a list of categoricals

categoricals = list()

for x in df.columns:

    if df[x].dtype == 'object':

        categoricals.append(x)
df[categoricals].nunique()
def plot_val_counts(df, col=''):

    plt.figure(figsize=(5,5))

    plt.grid(True)

    plt.bar(df[col][df.Churn=='Yes'].value_counts().index, 

            df[col][df.Churn=='Yes'].value_counts().values)

    plt.title(f'{col}')

    plt.xticks(rotation=-90)
plot_val_counts(df, col='HandsetPrice')
plot_val_counts(df, col='CreditRating')
plot_val_counts(df, col='Occupation')
plot_val_counts(df, col='PrizmCode')
def plot_distro(df, col = '', y_limit=None, x_limit_r=None, x_limit_l = None):

    plt.figure(figsize=(10,10))

    plt.grid(True)

    sns.distplot(df[col][df.Churn == 'Yes'])

    sns.distplot(df[col][df.Churn == 'No'])

    plt.legend(['churn_flag_yes', 'churn_flag_no'])

    if y_limit:

        plt.ylim(top=y_limit)

    if x_limit_r:

        plt.xlim(right=x_limit_r)

    if x_limit_l:

        plt.xlim(left=x_limit_l)
plot_distro(df, col='PercChangeMinutes', x_limit_r=1200, x_limit_l=-1200)
plot_distro(df, col='TotalRecurringCharge', x_limit_r=180)
plot_distro(df, col='DirectorAssistedCalls', y_limit=.3, x_limit_r=10)
plot_distro(df, col='MonthlyRevenue', x_limit_r=200)
plt.figure(figsize=(10,10))

plt.grid(True)

sns.boxplot(x=df.Occupation[df.Churn == 'Yes'], y=df.MonthlyRevenue[df.Churn == 'Yes'])

#sns.boxplot(x=df.Occupation[df.Churn == 'No'], y=df.MonthlyRevenue[df.Churn == 'No'])
plt.figure(figsize=(10,10))

plt.grid(True)

sns.boxplot(x=df.Occupation, y=df.MonthlyRevenue, hue=df.Churn)

plt.ylim(top=100)
plt.figure(figsize=(10,10))

plt.grid(True)

sns.boxplot(x=df.ChildrenInHH, y=df.MonthlyRevenue, hue=df.Churn)

#sns.boxplot(x=df.ChildrenInHH[df.Churn == 'No'], y=df.MonthlyRevenue[df.Churn == 'No'])

plt.ylim(top=150)
df.MonthsInService.describe()
tenure_churn = df.MonthsInService[df.Churn == 'Yes'].value_counts()

tenure_no_churn = df.MonthsInService[df.Churn == 'No'].value_counts()
tenure = pd.merge(tenure_churn.reset_index(), tenure_no_churn.reset_index(), on='index')
tenure = tenure.sort_values(by='index')
tenure = tenure.reset_index().drop(columns='level_0')
tenure.columns
plt.figure(figsize=(10,10))

plt.grid(True)

sns.pointplot(x=tenure.index, y=tenure.MonthsInService_x, color='red')

sns.pointplot(x=tenure.index, y=tenure.MonthsInService_y, color='green')

plt.xticks(rotation=90)

plt.title('When the churn picks')
def get_lists_of_dtypes(df):

    """

    Helper function to create list of features by type and by number of unique

    values they consist of.

    """

    strings = list()

    integers = list()

    floats = list()

    # Checking for partial string match to append accordingly value type

    # As here we might have different type of ints and floats

    # Note that strings we're returning as dictionary, to have number of unique vals for each feature

    for x in df.columns[2:]:

        if str(df[x].dtype)[:3] in 'obj':

            strings.append({x:len(df[x].unique())})

        elif str(df[x].dtype)[:3] in 'int':

            integers.append(x)

        elif str(df[x].dtype)[:3] in 'flo':

            floats.append(x)

        else:

            continue

    return strings,integers, floats
s, i, f = get_lists_of_dtypes(df)
s
def prep_categorical_features(s):

    """

    helper function to return features that we want to one hot encode

    """

    one_hot = list()

    binary = list()

    for x in s:

        for k, v in x.items():

            if v > 2:

                one_hot.append(k)

            else:

                binary.append(k)

    return one_hot, binary
one_hot, binary = prep_categorical_features(s)
def pairwise(col_1, col_2):

    """

    calculates pairwise features

    for given two dataframe columns

    """

    tot = col_1 + col_2

    diff = col_1 - col_2

    ratio = col_1/col_2

    return tot, diff, ratio
def stats(col):

    """

    calculates stats for given

    dataframe column

    """

    mini = col.min()

    maxi = col.max()

    avg = col.mean()

    return mini, maxi, avg
def feature_engine_numericals(dff, i, f):

    """

    Expands dataframe based on current lists of

    numerical features (int, floats)

    """

    numericals = i + f

    df = dff.copy()

    for x in numericals:

        df[f'{x}_min'], df[f'{x}_max'], df[f'{x}_mean'] = stats(df[x])

        for e in numericals:

            if e==x:

                pass

            else:

                df[f'sum_{x}_{e}'], df[f'diff_{x}_{e}'], df[f'ratio_{x}_{e}'] = pairwise(df[x], df[e])

    return df
%%time

pair_df = feature_engine_numericals(df, i, f)
def feature_engine_categoricals(dff, binary, one_hot):

    """

    Function to expand dataframe by one-hot encoding

    categorical variables also, changes datatype to float

    """

    df = dff.copy()

    lb = LabelBinarizer()

    for b in binary:

        df[f'{b}_tr'] = lb.fit_transform(df[b]).astype(np.float64)

        df = df.drop(columns=b)

    df = pd.get_dummies(df, columns=one_hot, dtype=float)

    return df
%%time

pair_df = feature_engine_categoricals(pair_df, binary, one_hot)
pair_df.shape, df.shape
pair_df.head()
le = LabelEncoder()
lab = le.fit_transform(pair_df.Churn).astype(np.float64)
l = pd.DataFrame({'lbls':pair_df.Churn, 'l_tr':lab})
l.head()
feats = pair_df.iloc[:,2:]
x_train, x_test, y_train, y_test = train_test_split(feats, lab, test_size = .25, random_state = 7)
x_tr, x_ev, y_tr, y_ev = train_test_split(x_train, y_train, test_size = .05, random_state = 7)
x_tr.shape, x_ev.shape, y_tr.dtype, y_ev.dtype
train_data = lgbm.Dataset(data=x_tr, label=y_tr)

val_data = lgbm.Dataset(data=x_ev, label=y_ev)
# tuning copied from https://www.kaggle.com/avanwyk/a-lightgbm-overview

# Note that there is no param search here, as this is meant to be a base line model.



advanced_params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    

    'learning_rate': 0.1,

    'num_leaves': 141, # more leaves increases accuracy, but may lead to overfitting.

    

    'max_depth': 7, # the maximum tree depth. Shallower trees reduce overfitting.

    'min_split_gain': 0, # minimal loss gain to perform a split

    'min_child_samples': 21, # or min_data_in_leaf: specifies the minimum samples per leaf node.

    'min_child_weight': 5, # minimal sum hessian in one leaf. Controls overfitting.

    

    'lambda_l1': 0.5, # L1 regularization

    'lambda_l2': 0.5, # L2 regularization

    

    'feature_fraction': 0.7, # randomly select a fraction of the features before building each tree.

    # Speeds up training and controls overfitting.

    'bagging_fraction': 0.5, # allows for bagging or subsampling of data to speed up training.

    'bagging_freq': 0, # perform bagging on every Kth iteration, disabled if 0.

    

    'scale_pos_weight': 99, # add a weight to the positive class examples (compensates for imbalance).

    

    'subsample_for_bin': 200000, # amount of data to sample to determine histogram bins

    'max_bin': 1000, # the maximum number of bins to bucket feature values in.

    # LightGBM autocompresses memory based on this value. Larger bins improves accuracy.

    

    'nthread': 4, # number of threads to use for LightGBM, best set to number of actual cores.

}
# train function from https://www.kaggle.com/avanwyk/a-lightgbm-overview

def train_gbm(params, training_set, validation_set, init_gbm=None, boost_rounds=100, early_stopping_rounds=0, metric='auc'):

    evals_result = {} 



    gbm = lgbm.train(params, # parameter dict to use

                    training_set,

                    init_model=init_gbm, # initial model to use, for continuous training.

                    num_boost_round=boost_rounds, # the boosting rounds or number of iterations.

                    early_stopping_rounds=early_stopping_rounds, # early stopping iterations.

                    # stop training if *no* metric improves on *any* validation data.

                    valid_sets=validation_set,

                    evals_result=evals_result, # dict to store evaluation results in.

                    verbose_eval=True) # print evaluations during training.

    

    return gbm, evals_result
gbm, evals_result = train_gbm(advanced_params, training_set=train_data, validation_set=val_data,

                             boost_rounds=1000, early_stopping_rounds=50)
y_hat = gbm.predict(x_test)
test_res = pd.DataFrame({'y_true':y_test, 'y_hat':y_hat})
test_res.y_hat[test_res.y_true == 0].shape, test_res.y_hat[test_res.y_true == 1].shape
roc_auc_score(test_res.y_true, test_res.y_hat)
test_res.y_hat[test_res.y_true == 0].describe()
test_res.y_hat[test_res.y_true == 1].describe()
def plot_distro(df, col = '', fiter_col = '', y_limit=None, x_limit_r=None, x_limit_l = None):

    plt.figure(figsize=(10,10))

    plt.grid(True)

    sns.distplot(df[col][df[fiter_col] == 1])

    sns.distplot(df[col][df[fiter_col] == 0])

    plt.legend(['churn_flag_yes', 'churn_flag_no'])

    if y_limit:

        plt.ylim(top=y_limit)

    if x_limit_r:

        plt.xlim(right=x_limit_r)

    if x_limit_l:

        plt.xlim(left=x_limit_l)
plot_distro(test_res, col = 'y_hat', fiter_col = 'y_true', y_limit=None, x_limit_r=None, x_limit_l = .8)
plt.figure(figsize=(10,10))

plt.grid(True)

sns.distplot(np.exp(test_res.y_hat[test_res.y_true == 0]), color='green')

sns.distplot(np.exp(test_res.y_hat[test_res.y_true == 1]), color='red')

plt.title('Distribution of the results, for two classes')

plt.legend(['no_churn', 'churn'])
plt.figure(figsize=(12,12))

plt.grid(True)

sns.distplot(1/np.log(test_res.y_hat[test_res.y_true == 0]), color='green')

sns.distplot(1/np.log(test_res.y_hat[test_res.y_true == 1]), color='red')

plt.plot([-36.3, -36.3], [0, 0.024], 'bo--', linewidth=2.5)

plt.plot([-45, -45], [0, 0.024], 'go--', linewidth=2.5)

plt.title('Distribution of the results, for two classes with upper thresholds')



plt.legend(['best_auc_threshold','threshold_business', 'no_churn', 'churn'])
1/np.log(test_res['y_hat'][test_res['y_true'] == 1]).describe()
test_res['y_transformed'] = 1/np.log(test_res['y_hat'])
def plot_roc_curve(test_res, threshold = -39):

    ns_probs = [0 for _ in range(len(test_res))]

    fpr, tpr, threshold = roc_curve(test_res.y_true, np.where(test_res.y_transformed < threshold, 1, 0))

    _fpr_, _tpr_, _threshold_ = roc_curve(test_res.y_true, ns_probs)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10,10))

    plt.grid(True)

    plt.title("ROC Curve. Area under Curve: {:.3f}".format(roc_auc))

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    _ = plt.plot(fpr, tpr, 'r')

    __ = plt.plot(_fpr_, _tpr_, 'b', ls = '--' )
plot_roc_curve(test_res, -36.3)
plot_roc_curve(test_res, -45)
test_res.y_transformed
lgbm.plot_importance(gbm, figsize=(10,12), max_num_features=15,importance_type='split' )
lgbm.plot_importance(gbm, figsize=(10,12), max_num_features=15,importance_type='gain' )
def plot_conf_mat(cm):

    """

    Helper function to plot confusion matrix.

    With text centerred.

    """

    plt.figure(figsize=(8,8))

    ax = sns.heatmap(cm, annot=True,fmt="d",annot_kws={"size": 16})

    bottom, top = ax.get_ylim()

    ax.set_ylim(bottom + 0.5, top - 0.5)
# AUC

cm_auc = confusion_matrix(test_res.y_true, np.where(test_res.y_transformed < -36.3, 1, 0), labels=[0, 1])
plot_conf_mat(cm_auc)
# Business

cm_bus = confusion_matrix(test_res.y_true, np.where(test_res.y_transformed < -43, 1, 0), labels=[0, 1])
plot_conf_mat(cm_bus)
# Our testing population

test_res.y_true.value_counts()
test_res.head()
# first threshold to explore is AUC oriented -36.3

auc_based = test_res[test_res.y_transformed <= -36.3]

auc_based.y_true.value_counts()
auc_based.y_true.value_counts()/auc_based.shape[0]
# second threshold to explore would be Business oriented -43

bus_based = test_res[test_res.y_transformed <= -43]

bus_based.y_true.value_counts()
bus_based.y_true.value_counts()/bus_based.shape[0]
df.MonthlyRevenue.describe()
# average per month and retention rate

avg_pm = 58.5

ret_rate = .6
def clv(avg_pm, ret_rate):

    """

    Example calculation of CLV per year

    with assumed retention rate of customers.

    """

    clv = 12 * avg_pm / (ret_rate/(1-ret_rate))

    return clv
cust_lv = clv(avg_pm, ret_rate)
cust_lv
# so we will have 4000 customers with auc model

# and 2650 from business model

# retention rate would be the same

new_avg = 58.5 * .9
new_avg
campaign_clv = clv(new_avg, ret_rate)
campaign_clv
auc_campaing = 4000 * campaign_clv

bus_campaign = 2650 * campaign_clv
auc_campaing, bus_campaign
pr = precision_score(test_res.y_true, np.where(test_res.y_transformed < -43, 1, 0))

pr
acc = accuracy_score(test_res.y_true, np.where(test_res.y_transformed < -43, 1, 0))

acc
precision_score(test_res.y_true, np.where(test_res.y_transformed < -36.3, 1, 0))
accuracy_score(test_res.y_true, np.where(test_res.y_transformed < -36.3, 1, 0))
f1_score(test_res.y_true, np.where(test_res.y_transformed < -43, 1, 0))
f1_score(test_res.y_true, np.where(test_res.y_transformed < -36.3, 1, 0))