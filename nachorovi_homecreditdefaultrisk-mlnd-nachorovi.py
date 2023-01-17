import numpy as np # linear algebra
import pandas as pd # data processing, CSVfile I/O (e.g. pd.read_csv)
import os
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
from plotly import tools
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
from sklearn.ensemble import RandomForestRegressor
# http://lightgbm.readthedocs.io/en/latest/Python-Intro.html
# https://github.com/Microsoft/LightGBM
import lightgbm as lgb
# Add evaluation metric to measure the model's performance
# Regression metrics available:
# http://scikit-learn.org/stable/modules/classes.html#regression-metrics
# http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
# http://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
# Cannot use sklearn.metrics.accuracy_score as it is a Classification metric
from sklearn.metrics import make_scorer, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
import itertools

#warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
# Support functions
def bar_hor(df, col, title, color, w=None, h=None, lm=0, limit=100, return_trace=False, rev=False, xlb = False):
    cnt_srs = df[col].value_counts()
    yy = cnt_srs.head(limit).index[::-1] 
    xx = cnt_srs.head(limit).values[::-1] 
    if rev:
        yy = cnt_srs.tail(limit).index[::-1] 
        xx = cnt_srs.tail(limit).values[::-1] 
    if xlb:
        trace = go.Bar(y=xlb, x=xx, orientation = 'h', marker=dict(color=color))
    else:
        trace = go.Bar(y=yy, x=xx, orientation = 'h', marker=dict(color=color))
    if return_trace:
        return trace 
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

def bar_hor_noagg(x, y, title, color, w=None, h=None, lm=0, limit=100, rt=False):
    trace = go.Bar(y=x, x=y, orientation = 'h', marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def bar_ver_noagg(x, y, title, color, w=None, h=None, lm=0, rt = False):
    trace = go.Bar(y=y, x=x, marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
def gp(col, title):
    df1 = app_train[app_train["TARGET"] == 1]
    df0 = app_train[app_train["TARGET"] == 0]
    a1 = df1[col].value_counts()
    b1 = df0[col].value_counts()
    
    total = dict(app_train[col].value_counts())
    x0 = a1.index
    x1 = b1.index
    
    y0 = [float(x)*100 / total[x0[i]] for i,x in enumerate(a1.values)]
    y1 = [float(x)*100 / total[x1[i]] for i,x in enumerate(b1.values)]

    trace1 = go.Bar(x=a1.index, y=y0, name='Target : 1', marker=dict(color="#96D38C"))
    trace2 = go.Bar(x=b1.index, y=y1, name='Target : 0', marker=dict(color="#FEBFB3"))
    return trace1, trace2 
# This implementation was copied from: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
# List available data files
#print(os.listdir("../input"))
print("Loading data files...")

start = time()
# Load all the datasets and reduce the memory usage
posc_bal = reduce_mem_usage(pd.read_csv("../input/POS_CASH_balance.csv"))
bureau_bal = reduce_mem_usage(pd.read_csv("../input/bureau_balance.csv"))
app_train = reduce_mem_usage(pd.read_csv("../input/application_train.csv"))
prev_app = reduce_mem_usage(pd.read_csv("../input/previous_application.csv"))
inst_pay = reduce_mem_usage(pd.read_csv("../input/installments_payments.csv"))
cc_bal = reduce_mem_usage(pd.read_csv("../input/credit_card_balance.csv"))
app_test = reduce_mem_usage(pd.read_csv("../input/application_test.csv"))
bureau = reduce_mem_usage(pd.read_csv("../input/bureau.csv"))
end = time()

print("Finished loading data files and running memory optimization in {} seconds.".format(int(round(end - start))))
# Show first 5 rows of each dataset
print('Point of Sale Cash Balance')
display(posc_bal.head())
print('Buereau Balance')
display(bureau_bal.head())
print('Applications Train')
display(app_train.head())
print('Previous Applications')
display(prev_app.head())
print('Installment Payments')
display(inst_pay.head())
print('Credit Card Balance')
display(cc_bal.head())
print('Applications Test')
display(app_test.head())
print('Bureau')
display(bureau.head())
# Show dataset descriptive statistics
print('Point of Sale Cash Balance')
display(posc_bal.describe(exclude=['category']))
print('Buereau Balance')
display(bureau_bal.describe(exclude=['category']))
print('Applications Train')
display(app_train.describe(exclude=['category']))
print('Previous Applications')
display(prev_app.describe(exclude=['category']))
print('Installment Payments')
display(inst_pay.describe(exclude=['category']))
print('Credit Card Balance')
display(cc_bal.describe(exclude=['category']))
print('Applications Test')
display(app_test.describe(exclude=['category']))
print('Bureau')
display(bureau.describe(exclude=['category']))
eda = pd.DataFrame(
    [
        ['Point of Sale Cash Balance', posc_bal.shape[0], posc_bal.shape[1] - 2, np.sum(posc_bal.dtypes=='category'), 
            np.sum(posc_bal.isnull().sum() > 0), posc_bal.isnull().sum().sum()], # Features don't include SK_ID_PREV and SK_ID_CURR
        ['Bureau Balance', bureau_bal.shape[0], bureau_bal.shape[1] - 1, np.sum(bureau_bal.dtypes=='category'), 
            np.sum(bureau_bal.isnull().sum() > 0), bureau_bal.isnull().sum().sum()], # Features don't include SK_ID_BUREAU
        ['Applications Train', app_train.shape[0], app_train.shape[1] - 2, np.sum(app_train.dtypes=='category'), 
            np.sum(app_train.isnull().sum() > 0), app_train.isnull().sum().sum()], # Features don't include SK_ID_CURR or TARGET
        ['Previous Applications', prev_app.shape[0], prev_app.shape[1] - 2, np.sum(prev_app.dtypes=='category'), 
            np.sum(prev_app.isnull().sum() > 0), prev_app.isnull().sum().sum()], # Features don't include SK_ID_PREV and SK_ID_CURR
        ['Installment Payments', inst_pay.shape[0], inst_pay.shape[1] - 2, np.sum(inst_pay.dtypes=='category'), 
            np.sum(inst_pay.isnull().sum() > 0), inst_pay.isnull().sum().sum()], # Features don't include SK_ID_PREV and SK_ID_CURR
        ['Credit Card Balance', cc_bal.shape[0], cc_bal.shape[1] - 2, np.sum(cc_bal.dtypes=='category'), 
            np.sum(cc_bal.isnull().sum() > 0), cc_bal.isnull().sum().sum()], # Features don't include SK_ID_PREV and SK_ID_CURR
        ['Applications Test', app_test.shape[0], app_test.shape[1] - 1, np.sum(app_test.dtypes=='category'), 
            np.sum(app_test.isnull().sum() > 0), app_test.isnull().sum().sum()], # Features don't include SK_ID_CURR
        ['Bureau', bureau.shape[0], bureau.shape[1] - 2, np.sum(bureau.dtypes=='category'), 
            np.sum(bureau.isnull().sum() > 0), bureau.isnull().sum().sum()], # Features don't include SK_ID_CURR and SK_ID_BUREAU
    ],
    columns=['Dataset', 'samples', 'number_features', 'number_categorical_features', 'number_features_missing_values', 
                'total_number_missing_values']
)

display(eda.head(8))
plt.figure(figsize=(18,9))
plt.subplot(121)
app_train["TARGET"].value_counts().plot(fontsize = 16,
                                        kind = 'pie',
                                        autopct = "%1.0f%%",
                                        colors = sns.color_palette("prism",8),
                                        startangle = 90,
                                        labels=["1 - Repayer","0 - Defaulter"],
                                        explode=[.1,0],
                                       )
plt.title("Distribution of Target Label for Applications Train dataset", fontsize=20)
# Implementation source: https://www.kaggleusercontent.com/kf/4442153/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Df_QZcauc2BVOOmdHRjw1Q.ruvOVG8p44cAqUgN2tZHTPK-y8DwzYtkIoGA39JWR938aOHRdCqQRYjQj1U8AAiXqRfoRScRMjXH_DrMDqBWO9JIBKjTxS7yQyC3ouVc-MuExzzH0lGZdfJT2HJGkjvqSVLm4gYg7ML3r_jmJ3dP--6dmgHGsW1TQ6D04GnZzk6xwZseKGjCzeIYavlz44Qj.WDYyfq5ILj9HsKasnQ37uA/__results__.html#Comparing-summary-statistics-between-defaulters-and-non---defaulters-for-loan-amounts-.
cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']

df = app_train.groupby("TARGET")[cols].describe().transpose().reset_index()
display(df)
df = df[df["level_1"].isin([ 'mean', 'std', 'min', 'max'])] 
df_x = df[["level_0","level_1",0]]
df_y = df[["level_0","level_1",1]]
df_x = df_x.rename(columns={'level_0':"amount_type", 'level_1':"statistic", 0:"amount"})
df_x["type"] = "1 - Repayer"
df_y = df_y.rename(columns={'level_0':"amount_type", 'level_1':"statistic", 1:"amount"})
df_y["type"] = "0 - Defaulter"
df_new = pd.concat([df_x,df_y],axis = 0)

stat = df_new["statistic"].unique().tolist()
length = len(stat)

plt.figure(figsize=(13,15))

for i,j in itertools.zip_longest(stat,range(length)):
    plt.subplot(2,2,j+1)
    fig = sns.barplot(df_new[df_new["statistic"] == i]["amount_type"],df_new[df_new["statistic"] == i]["amount"],
                hue=df_new[df_new["statistic"] == i]["type"],palette=["g","r"])
    plt.title(i + "--Defaulters vs Non defaulters")
    plt.subplots_adjust(hspace = .4)
    fig.set_facecolor("lightgrey")
plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_BIRTH")
ax = sns.distplot(app_train["DAYS_BIRTH"])
plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
ax = sns.distplot(app_train["AMT_CREDIT"])
plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_ID_PUBLISH")
ax = sns.distplot(app_train["DAYS_ID_PUBLISH"])
plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_REGISTRATION")
ax = sns.distplot(app_train["DAYS_REGISTRATION"])
plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_EMPLOYED")
ax = sns.distplot(app_train["DAYS_EMPLOYED"])
# Merge training and testing datasets - This will help in two ways:
# When handling categorical variables it will ensure both datasets end up with the same features
# When handling missing values, if we use the mean to fill in missing values, they will be more representative
app_train['is_train'] = 1
app_train['is_test'] = 0
app_test['is_train'] = 0
app_test['is_test'] = 1
print("\nJoining the training(app_train) and testing(app_test) dataset for pre-processing into pandas DataFrame 'data'.")

# data = pd.concat([app_train, app_test], axis=0, sort=False)
# ERROR: TypeError: concat() got an unexpected keyword argument 'sort'
data = pd.concat([app_train, app_test], axis=0)
# Substract 4 from the features count for the columns 'TARGET', 'SK_ID_CURR', 'is_train', 'is_test' for app_train
# And substract 3 for app_test, as it doesn't have a 'TARGET' column
print("app_train has {0:,} samples and {1} features.".format(app_train.shape[0], app_train.shape[1]-4))
print("app_test has {0:,} samples and {1} features.".format(app_test.shape[0], app_test.shape[1]-3))
print("data has {0:,} samples and {1} features BEFORE one-hot encoding.".format(data.shape[0], data.shape[1]-4))
assert(data.shape[0] == app_train.shape[0] + app_test.shape[0])
assert(data.shape[1] >= max(app_train.shape[1], app_test.shape[1]))
# Handle Categorical variables - Turn categorical variables into numerical features using the one-hot encoding scheme
# Support function for one-hot encoding
def _one_hot_encoding(data):
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
    return pd.get_dummies(data)

# Handle categorical variables
print("\nPerforming one-hot encoding on {} dataset.".format('data'))
data = _one_hot_encoding(data)
# Substract 4 from the features count for the columns 'TARGET', 'SK_ID_CURR', 'is_train', 'is_test'
print("app has {0:,} samples and {1} features AFTER one-hot encoding.".format(data.shape[0], data.shape[1]-4))
posc_bal = _one_hot_encoding(posc_bal)
#bureau_bal = _one_hot_encoding(bureau_bal)
prev_app = _one_hot_encoding(prev_app)
inst_pay = _one_hot_encoding(inst_pay)
cc_bal = _one_hot_encoding(cc_bal)
bureau = _one_hot_encoding(bureau)
# Keep a copy of the application_train & application_test datasets without merging with the rest of the datasets
data_train_test = data.copy()
# Merge Point of Sale Cash Balance dataset
print("Merge 'Point of Sale Cash Balance' dataset.")
# Count the number of previous applications for a given 'SK_ID_CURR', and create a new feature
posc_bal_count = posc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
posc_bal['POSC_BAL_COUNT'] = posc_bal['SK_ID_CURR'].map(posc_bal_count['SK_ID_PREV'])
# Remove the 'SK_ID_PREV' column from the dataset as it doesn't add value
posc_bal = posc_bal.drop(['SK_ID_PREV'], axis=1)

# Average values for all other features in previous applications
posc_bal_avg = posc_bal.groupby('SK_ID_CURR').mean()
posc_bal_avg.columns = ['pcb_' + col for col in posc_bal_avg.columns]
data = data.merge(right=posc_bal_avg.reset_index(), how='left', on='SK_ID_CURR')
'''
# Merge Bureau Balance dataset
print("Merge 'Bureau Balance' dataset.")
#'SK_ID_BUREAU'
# Count the number of previous applications for a given 'SK_ID_CURR', and create a new feature
bureau_bal_count = bureau_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
bureau_bal['bureau_bal_COUNT'] = bureau_bal['SK_ID_CURR'].map(bureau_bal_count['SK_ID_PREV'])
# Remove the 'SK_ID_PREV' column from the dataset as it doesn't add value
bureau_bal = bureau_bal.drop(['SK_ID_PREV'], axis=1)

# Average values for all other features in previous applications
bureau_bal_avg = bureau_bal.groupby('SK_ID_CURR').mean()
bureau_bal_avg.columns = ['posc_' + col for col in bureau_bal_avg.columns]
data_train = data_train.merge(right=bureau_bal_avg.reset_index(), how='left', on='SK_ID_CURR')
'''
# Merge Previous Applications dataset
print("Merge 'Previous Applications' dataset.")
# Count the number of previous applications for a given 'SK_ID_CURR'
prev_app_count = prev_app[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
prev_app['PREV_COUNT'] = prev_app['SK_ID_CURR'].map(prev_app_count['SK_ID_PREV'])
# Remove the 'SK_ID_PREV' column from the dataset as it doesn't add value
prev_app = prev_app.drop(['SK_ID_PREV'], axis=1)

# Average values for all other features in previous applications
prev_app_avg = prev_app.groupby('SK_ID_CURR').mean()
prev_app_avg.columns = ['pa_' + col for col in prev_app_avg.columns]
data = data.merge(right=prev_app_avg.reset_index(), how='left', on='SK_ID_CURR')
# Merge Installments Payments dataset
print("Merge 'Installments Payments' dataset.")
# Count the number of installments payments for a given 'SK_ID_CURR', and create a new feature
inst_pay_count = inst_pay[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
inst_pay['INST_PAY_COUNT'] = inst_pay['SK_ID_CURR'].map(inst_pay_count['SK_ID_PREV'])
# Remove the 'SK_ID_PREV' column from the dataset as it doesn't add value
inst_pay = inst_pay.drop(['SK_ID_PREV'], axis=1)

## Average values for all other features in previous applications
inst_pay_avg = inst_pay.groupby('SK_ID_CURR').mean()
inst_pay_avg.columns = ['ip_' + col for col in inst_pay_avg.columns]
data = data.merge(right=inst_pay_avg.reset_index(), how='left', on='SK_ID_CURR')
# Merge Credit Card Balance dataset
print("Merge 'Credit Card Balance' dataset.")
# Count the number of previous applications for a given 'SK_ID_CURR', and create a new feature
cc_bal_count = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
cc_bal['CC_BAL_COUNT'] = cc_bal['SK_ID_CURR'].map(cc_bal_count['SK_ID_PREV'])
# Remove the 'SK_ID_PREV' column from the dataset as it doesn't add value
cc_bal = cc_bal.drop(['SK_ID_PREV'], axis=1)

## Average values for all other features in previous applications
cc_bal_avg = cc_bal.groupby('SK_ID_CURR').mean()
cc_bal_avg.columns = ['ccb_' + col for col in cc_bal_avg.columns]
data = data.merge(right=cc_bal_avg.reset_index(), how='left', on='SK_ID_CURR')
# Merge Bureau dataset
print("Merge 'Bureau' dataset.")
# Count the number of credits registered in the bureau for a given 'SK_ID_CURR', and create a new feature
bureau_count = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
bureau['BUREAU_COUNT'] = bureau['SK_ID_CURR'].map(bureau_count['SK_ID_BUREAU'])
# Remove the 'SK_ID_BUREAU' column from the dataset as it doesn't add value
bureau = bureau.drop(['SK_ID_BUREAU'], axis=1)

## Average values for all other features in previous applications
bureau_avg = bureau.groupby('SK_ID_CURR').mean()
bureau_avg.columns = ['b_' + col for col in bureau_avg.columns]
data = data.merge(right=bureau_avg.reset_index(), how='left', on='SK_ID_CURR')
# Transforming skewed continuous features
#skewed = ['DAYS_EMPLOYED']
#data[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
# I need to handle negative numbers, if x = -1 then it will throw an error; log(0) = Inf
plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_EMPLOYED")
ax = sns.distplot(data["DAYS_EMPLOYED"])
# Normalizing numerical features
from sklearn.preprocessing import MinMaxScaler
#app_train_copy = app_train.copy()

scaler = MinMaxScaler()
# Full list of top ten features, discounting EXT_SOURCE_X becuase they are already normalizaed:
# ['DAYS_BIRTH', 'AMT_ANNUITY', 'AMT_CREDIT', 'DAYS_ID_PUBLISH', 'pcb_CNT_INSTALMENT_FUTURE', 'DAYS_REGISTRATION', 'DAYS_EMPLOYED']

# numerical = ['DAYS_BIRTH', 'AMT_ANNUITY', 'AMT_CREDIT', 'DAYS_ID_PUBLISH']
# 12 entries in 'AMT_ANNUITY' are NaN - I need to fix that first before Normalizing

# 'pcb_CNT_INSTALMENT_FUTURE' belongs to a different dataset

# numerical = ['DAYS_BIRTH', 'AMT_CREDIT', 'DAYS_ID_PUBLISH']

numerical = ['DAYS_BIRTH', 'AMT_CREDIT', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_EMPLOYED']
data[numerical] = scaler.fit_transform(data[numerical])
data_to_use = 'ALL' # 'ALL' or 'data_train_test'
if data_to_use == 'data_train_test':
    data = data_train_test.copy()
# Handle missing data
# https://pandas.pydata.org/pandas-docs/stable/missing_data.html#filling-with-a-pandasobject
# https://www.kaggle.com/dansbecker/handling-missing-values
# http://scikit-learn.org/dev/modules/generated/sklearn.impute.SimpleImputer.html
print("\nFilling NaN values in the dataset using pandas.fillna() using the column mean() value.")
print("Number of NaN values in the dataset BEFORE running pandas.fillna(): {:,}".format(data.isnull().sum().sum()))
data = data.fillna(data.mean())
nan_after = data.isnull().sum().sum()
print("Number of NaN values in the dataset AFTER running pandas.fillna(): {:,}".format(nan_after))
assert(nan_after == 0)
import gc
# Clean variables that are no longer needed
# Not used yet: bureau_bal_count, bureau_bal_avg
del posc_bal, posc_bal_count, posc_bal_avg, bureau_bal, app_train, app_test
del prev_app, prev_app_count, prev_app_avg, inst_pay, inst_pay_count, inst_pay_avg, cc_bal, cc_bal_count, cc_bal_avg
del bureau, bureau_count, bureau_avg, data_train_test
gc.collect()
# Separate the data into the original test and training datasets
# Remove columns 'TARGET', 'SK_ID_CURR', 'is_train', 'is_test' as they are not features
print("\nSeparating the training and testing dataset after completing pre-processing.")
train = data[data['is_train'] == 1]

# Separate the 'target label' from the training dataset
target = train['TARGET']
train = train.drop(['TARGET', 'SK_ID_CURR', 'is_test', 'is_train'], axis=1)
test = data[data['is_test'] == 1]

# To be used when preparing the submission
test_id = test['SK_ID_CURR']
test = test.drop(['TARGET', 'SK_ID_CURR', 'is_test', 'is_train'], axis=1)
print("train has {:,} samples and {} features.".format(train.shape[0], train.shape[1]))
print("test has {:,} samples and {} features.".format(test.shape[0], test.shape[1]))
# Split 'features' and 'target label' data into training and validation data using train_test_split
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
print("\nSplitting the training dataset into actual training and validation datasets")
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=42)
assert(train.shape[0] == X_train.shape[0] + X_val.shape[0])
assert(X_train.shape[1] == train.shape[1])
assert(X_val.shape[1] == train.shape[1])
assert(target.shape[0] == y_train.shape[0] + y_val.shape[0])
print("training dataset has {0:,} samples and {1} features.".format(X_train.shape[0], X_train.shape[1]))
print("validating dataset has {0:,} samples and {1} features.".format(X_val.shape[0], X_val.shape[1]))
# Run GridSearchCV or fully train an estimator
# 'grid_search_RFR', 'grid_search_LGBM', 'train_estimators', 'train_estimator_LGBM', 'train_estimator_RFR', 'LGBM_KFold'
run_mode = 'LGBM_KFold'
# Run GridSearchCV on LGBM
if run_mode == 'grid_search_LGBM':
    perc_samples = 0.15
    print("\nPreparing to run Hyperparameters tunning with GridSearchCV using {0:.2f}% of the training samples".format(perc_samples * 100))
    X_train_small = X_train[:int(perc_samples * X_train.shape[0])]
    y_train_small = y_train[:int(perc_samples * y_train.shape[0])]
    X_val_small = X_val[:int(perc_samples * X_val.shape[0])]
    y_val_small = y_val[:int(perc_samples * y_val.shape[0])]
    
    estimator = lgb.LGBMClassifier(
          objective='binary',
          metric='auc',
          num_iteration=5000, # num_boost_round=5000,
          verbose=1,
          silent=False,
          colsample_bytree=.8,
          subsample=.9,
          reg_alpha=.1,
          reg_lambda=.1,
          min_split_gain=.01,
          min_child_weight=1,
          # early_stopping_rounds=100
          # ValueError: For early stopping, at least one dataset and eval metric is required for evaluation
    )
    
    '''
    parameters = {
          'task': ['train'],
          'boosting_type': ['gbdt'],
          'objective': ['binary'],
          'metric': ['auc'],
          'learning_rate': [0.01],
          'num_leaves': [48],
          'num_iteration': [5000],
          'verbose': 0,
          'colsample_bytree': [.8],
          'subsample': [.9],
          'max_depth': [7],
          'reg_alpha': [.1],
          'reg_lambda': [.1],
          'min_split_gain': [.01],
          'min_child_weight': [1]
        }
    '''
    parameters = {
          'boosting_type': ['gbdt'], # 'dart'
          'num_leaves': [35, 48, 80],
          'min_data_in_leaf': [20], # [15, 20, 25],
          'learning_rate': [0.005],
          'max_depth': [7], # [6, 7, 8],
        }
    
    # Create a scorer to measure hyperparameters performance
    scorer = make_scorer(roc_auc_score)

    # Create GridSearchCV grid object
    grid_obj = GridSearchCV(estimator=estimator, 
                            param_grid=parameters, 
                            scoring=scorer)

    # Fit the GridSearchCV grid object with the reduced training dataset and find the best hyperparameters
    start = time()
    grid_fit = grid_obj.fit(X_train_small, y_train_small)
    end = time()
    grid_fit_time = (end - start) / 60 # Ellapsed time in minutes
    print("\nGridSearchCV estimator fit time: {0:.2f} minutes".format((end - start) / 60))

    print("\nPreparing to run Hyperparameters tunning with GridSearchCV using {0:.2f}% of the training samples".format(perc_samples * 100))
    print("\nParameters used for tunning: \n{}".format(parameters))
    # Get the best estimator
    best_est = grid_obj.best_estimator_
    print("\nBest Estimator: \n{}\n".format(best_est))

    # Get the best score
    best_score = grid_obj.best_score_
    print("\nBest Estimator Score: {}\n".format(best_score))

    # Get the best parameters
    best_params = grid_obj.best_params_
    print("\nBest Hyperparameters that yield the best score: \n{}\n".format(best_params))

    # Make predictions with unoptimized estimator on the validation set
    #pred_val = (estimator.fit(features_train_small, target_train_small)).predict(features_val_small)
    #print("\nUnoptimized Estimator prediction score on Validation set: \t{}".format(roc_auc_score(y_val_small, pred_val)))

    # Predict with the best estimator on the validation set
    best_pred_val = best_est.predict(X_val_small)
    print("\nOptimized Estimator prediction score on Validation set: \t{}".format(roc_auc_score(y_val_small, best_pred_val)))

# Run GridSearchCV
if run_mode == 'grid_search_RFR':
    perc_samples = 0.15
    print("\nPreparing to run Hyperparameters tunning with GridSearchCV using {0:.2f}% of the training samples".format(perc_samples * 100))
    features_train_small = X_train[:int(perc_samples * X_train.shape[0])]
    target_train_small = y_train[:int(perc_samples * y_train.shape[0])]
    features_val_small = X_val[:int(perc_samples * X_val.shape[0])]
    target_val_small = y_val[:int(perc_samples * y_val.shape[0])]
    #features_test_small = features_test[:int(perc_samples * features_test.shape[0])]

    # Initialize the Estimator (Learner or Regression Model)
    estimator = RandomForestRegressor(n_jobs=-1,
                                      random_state=42,
                                      verbose=0)

    # Determine which Parameters to tune
    '''
    Tested so far:
    parameters = {
        'n_estimators': [9, 10, 11, 12, 13, 14, 15],
        'criterion': ['mse', 'mae'],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7],
        'max_features': [0.01, 0.1, 0.25, 0.45, 0.5, 0.55, 0.6, 0.75],
        'min_samples_split': [2, 3, 4, 5],
        'warm_start': [False, True]
    }
    '''
    parameters = {
        'n_estimators': [130, 135, 145],
        'min_samples_leaf': [55, 62, 75],
        'max_features': [0.2], # [0.18, 0.2, 0.23]
        'min_samples_split': [2], # [2, 3]
    }

    # Create a scorer to measure hyperparameters performance
    scorer = make_scorer(roc_auc_score)

    # Create GridSearchCV grid object
    grid_obj = GridSearchCV(estimator=estimator, 
                            param_grid=parameters, 
                            scoring=scorer)

    # Fit the GridSearchCV grid object with the reduced training dataset and find the best hyperparameters
    start = time()
    grid_fit = grid_obj.fit(features_train_small, target_train_small)
    end = time()
    grid_fit_time = (end - start) / 60 # Ellapsed time in minutes
    print("\nGridSearchCV estimator fit time: {0:.2f} minutes".format((end - start) / 60))

    # Get the best estimator
    best_est = grid_obj.best_estimator_
    print("\nBest Estimator: \n{}\n".format(best_est))

    # Get the best score
    best_score = grid_obj.best_score_
    print("\nBest Estimator Score: {}\n".format(best_score))

    # Get the best parameters
    best_params = grid_obj.best_params_
    print("\nBest Hyperparameters that yield the best score: \n{}\n".format(best_params))

    # Make predictions with unoptimized estimator on the validation set
    pred_val = (estimator.fit(features_train_small, target_train_small)).predict(features_val_small)
    print("\nUnoptimized Estimator prediction score on Validation set: \t{}".format(roc_auc_score(target_val_small, pred_val)))

    # Predict with the best estimator on the validation set
    best_pred_val = best_est.predict(features_val_small)
    print("\nOptimized Estimator prediction score on Validation set: \t{}".format(roc_auc_score(target_val_small, best_pred_val)))

    # Predict with the best estimator on the testing set
    #pred_test = best_est.predict(features_test)
# Train estimator LGBM
if run_mode == 'train_estimator_LGBM':
    params = {
        'boosting_type': 'dart',
        'objective': 'binary',
        'learning_rate': 0.1,
        'min_data_in_leaf': 30,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.5,
        'scale_pos_weight': 2,
        'drop_rate': 0.02,
        'metric': 'auc',
        'num_boost_round': 200,
    }

    data_split = 'kfold' # Possible values: 'kfold' or 'train_test_split'
    if data_split == 'train_test_split':
        # Using split merged datasets with train_test_split
        lgb_train = lgb.Dataset(data=X_train, label=y_train)
        lgb_eval = lgb.Dataset(data=X_val, label=y_val)
        start = time()
        estimator = lgb.train(
            params = params,
            train_set = lgb_train,
            valid_sets = lgb_eval,
            early_stopping_rounds = 350,
            verbose_eval = 200
        )
        end = time()
    elif data_split == 'kfold':   
        # Using KFolds to split the merged dataset for cross-validation
        lgb_train_cv = lgb.Dataset(data=train, label=target)
        start = time()
        cv_results = lgb.cv(
            params = params,
            train_set = lgb_train_cv,
            nfold = 2,
            early_stopping_rounds = 50,
            stratified = True,
            verbose_eval = 50
        )
        optimum_boost_rounds = np.argmax(cv_results['auc-mean'])
        print('Optimum boost rounds = {}'.format(optimum_boost_rounds))
        print('Best LGBM CV result = {}'.format(np.max(cv_results['auc-mean'])))
        estimator = lgb.train(
            params = params,
            train_set = lgb_train_cv,
            num_boost_round = optimum_boost_rounds,
            verbose_eval = 50
        )
        end = time()
    
    print("\nEstimator fit time: {} seconds".format(int(round(end - start))))

    lgb.plot_importance(estimator, figsize=(12, 12), max_num_features=30);
print(estimator)
# Parameters from Aguiar https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code
# Train estimator LGBM with KFold Cross-validation
if run_mode == 'LGBM_KFold':
    from sklearn.model_selection import KFold, StratifiedKFold

    folds = KFold(n_splits=10, shuffle=True, random_state=1024)

    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test.shape[0])
    feature_importance = pd.DataFrame()
    feats = train.columns

    start = time()
    for n_fold, (train_index, valid_index) in enumerate(folds.split(train, target)):
        train_x, train_y = train.iloc[train_index], target.iloc[train_index]
        valid_x, valid_y = train.iloc[valid_index], target.iloc[valid_index]

        # LightGBM parameters found by Bayesian optimization
        clf = lgb.LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, 
        )

        clf.fit(
            train_x,
            train_y,
            eval_set = [(valid_x, valid_y)],
            eval_metric = 'auc',
            verbose = 200,
            early_stopping_rounds = 500,
        )

        oof_preds[valid_index] = clf.predict_proba(valid_x, num_iterations=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test, num_iterations=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance = pd.DataFrame()
        fold_importance['feature'] = feats
        fold_importance['importance'] = clf.feature_importances_
        fold_importance['fold'] = n_fold + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        print('Fold {:02d} AUC: {:.6f}'.format(n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_index])))

    end = time()
    print("\nEstimator fit time: {} seconds".format(int(round(end - start))))
    print('Full AUC score: {:.6f}'.format(roc_auc_score(target, oof_preds)))
# Train estimator LGBM with KFold Cross-validation
if run_mode == 'LGBM_KFold':
    # Display feature importance
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
# Train estimator LGBM with KFold Cross-validation
if run_mode == 'LGBM_KFold':
    # Prepare submission file
    submission = pd.DataFrame()
    submission['SK_ID_CURR'] = test_id
    submission['TARGET'] = sub_preds
    submission.to_csv('LGBM_SKFold.csv', index=False)
# Train estimator RandonForrestRegressor
if run_mode == 'train_estimator_RFR':
    # Initialize the Estimator (Learner or Regression Model) with the best hyperparameters
    # Alternative: n_estimators=135, max_features=0.2, min_samples_split=2, min_samples_leaf=62
    # Alternative2: criterion='mae', # default='mse', VERY SLOW
    estimator = RandomForestRegressor(n_estimators=125, # default=10
                                      max_features=0.2, # default='auto'
                                      min_samples_split=2, # default=2
                                      min_samples_leaf=75, # default=1
                                      n_jobs=-1, # default=1
                                      random_state=42, # default=None
                                      verbose=0) # default=0
    print("\nPreparing to train the following estimator: \n{}".format(estimator))

    # Fit the estimator with the training dataset
    start = time()
    estimator.fit(X_train, y_train)
    end = time()
    print("\nEstimator fit time: {} seconds".format(int(round(end - start))))

    # Predict with the validation dataset
    pred_val = estimator.predict(X_val)
    print("\nEstimator prediction score on Validation set: \t{}".format(roc_auc_score(y_val, pred_val)))
    
    # Determine the feature importance
    fi = pd.DataFrame()
    fi['feature'] = X_train.columns
    fi['importance'] = estimator.feature_importances_
    display(fi.sort_values(by=['importance'], ascending=False).head(10))

    # TODO: GRAPH THE FEATURE IMPORTANCE
if 'train_estimator_' in run_mode:
    # Predict using the 'test' dataset for submission
    pred_test = estimator.predict(test)
    
    # Prepare prediction for submission
    print("\nPreparing prediction for submission.")
    submission = pd.DataFrame()
    submission['SK_ID_CURR'] = test_id
    # Replace any negative number with zero, required for https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm/code
    # pred_test[pred_test < 0] = 0
    submission['TARGET'] = pred_test
    submission.head()
    file_name = run_mode.split('train_estimator_')[1] + '.csv'
    submission.to_csv(file_name, index=False)
