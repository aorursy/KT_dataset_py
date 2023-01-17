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
import seaborn as sns

pd.set_option('display.max_rows', 1000)
train = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')

test = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')
train
from sklearn.utils import shuffle

desired_apriori=0.05



# Get the indices per target value

idx_0 = train[train.target == 0].index

idx_1 = train[train.target == 1].index



# # Get original number of records per target value

nb_0 = len(train.loc[idx_0])

nb_1 = len(train.loc[idx_1])







# # Calculate the undersampling rate and resulting number of records with target=0

undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)

undersampled_nb_0 = int(undersampling_rate*nb_0)

print('Rate to undersample records with target=0: {}'.format(undersampling_rate))

print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))



# Randomly select records with target=0 to get at the desired a priori

undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)



# Construct list with remaining indices

idx_list = list(undersampled_idx) + list(idx_1)



# Return undersample data frame

train = train.loc[idx_list].reset_index(drop=True)
df = pd.concat((train.loc[:,'ps_ind_01':'ps_calc_20_bin'], test.loc[:,'ps_ind_01':'ps_calc_20_bin']))

df = df.replace(-1, np.NaN)
for column in df.columns:

    print(f"{column}: {df[column].nunique()}")

    if df[column].nunique() < 10:

        print(f"{df[column].value_counts()}")

    print("====================================")
def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)


# Dropping the variables with too many missing values

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']

df.drop(vars_to_drop, inplace=True,  axis=1)

col_to_drop = df.columns[df.columns.str.startswith('ps_calc_')]

df.drop(col_to_drop, inplace=True, axis=1)



def missing_value(df):

    col = df.columns

    for i in col:

        if df[i].isnull().sum()>0:

            df[i].fillna(df[i].mode()[0],inplace=True)

            

missing_value(df)



# for c in df.select_dtypes(include=['float64']).columns:

#     df[c]=df[c].astype(np.float32)

    

# for c in df.select_dtypes(include=['int64']).columns[2:]:

#     df[c]=df[c].astype(np.int8)

    



def category_type(df):

    col = df.columns

    for i in col:

        if (df[i].nunique()<=104) and (df[i].nunique()>2) :

            df[i] = df[i].astype('category')

category_type(df)



df.info()
tot_cat_col = list(df.select_dtypes(include=['category']).columns)

num_col = [c for c in df.columns if c not in tot_cat_col]
def descrictive_stat_feat(df):

    df = pd.DataFrame(df)

    dcol= [c for c in df.columns if df[c].nunique()>=10]

    d_median = df[dcol].median(axis=0)

    d_mean = df[dcol].mean(axis=0)

    q1 = df[dcol].apply(np.float32).quantile(0.25)

    q3 = df[dcol].apply(np.float32).quantile(0.75)

    

    #Add mean and median column to data set having more then 10 categories

    for c in dcol:

        df[c+str('_median_range')] = (df[c].astype(np.float32).values > d_median[c]).astype(np.int8)

        df[c+str('_mean_range')] = (df[c].astype(np.float32).values > d_mean[c]).astype(np.int8)

        df[c+str('_q1')] = (df[c].astype(np.float32).values < q1[c]).astype(np.int8)

        df[c+str('_q3')] = (df[c].astype(np.float32).values > q3[c]).astype(np.int8)

    return df



df = descrictive_stat_feat(df)
df.shape
df.head(3).transpose()
df.shape
def OHE(df,column):

    cat_col = column

    len_df = df.shape[0]

    c2,c3 = [],{}

    

    print('Categorical feature',len(column))

    for c in cat_col:

        if df[c].nunique()>2 :

            c2.append(c)

            c3[c] = 'ohe_'+c

    

    df = pd.get_dummies(df, prefix=c3, columns=c2, drop_first=True)

    return df

df = OHE(df,tot_cat_col)
def outlier(df,columns):

    for i in columns:

        quartile_1,quartile_3 = np.percentile(df[i],[25,75])

        quartile_f,quartile_l = np.percentile(df[i],[1,99])

        IQR = quartile_3-quartile_1

        lower_bound = quartile_1 - (1.5*IQR)

        upper_bound = quartile_3 + (1.5*IQR)

        print(i,lower_bound,upper_bound,quartile_f,quartile_l)

                

        df[i].loc[df[i] < lower_bound] = quartile_f

        df[i].loc[df[i] > upper_bound] = quartile_l

        

outlier(df,num_col)
df.shape
#creating matrices for feature selection:

X_train = df[:train.shape[0]]

X_test_fin = df[train.shape[0]:]

y = train.target

X_train['Y'] = y

df = X_train

df.head() ## DF for Model training
# #Correlation with output variable

# cor = df.corr()

# cor_target = (cor['Y'])

# #Selecting highly correlated features (8% level)

# relevant_features = cor_target[(cor_target<=-0.00) | (cor_target>=0.00) ]

# relevant_features.sort_values(ascending = False).head(1000)
def reduce_memory_usage(df):

    """ The function will reduce memory of dataframe

    Note: Apply this function after removing missing value"""

    intial_memory = df.memory_usage().sum()/1024**2

    print('Intial memory usage:',intial_memory,'MB')

    for col in df.columns:

        mn = df[col].min()

        mx = df[col].max()

        if df[col].dtype != object:            

            if df[col].dtype == int:

                if mn >=0:

                    if mx < np.iinfo(np.uint8).max:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < np.iinfo(np.uint16).max:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < np.iinfo(np.uint32).max:

                        df[col] = df[col].astype(np.uint32)

                    elif mx < np.iinfo(np.uint64).max:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)

            if df[col].dtype == float:

                df[col] =df[col].astype(np.float32)

    

    red_memory = df.memory_usage().sum()/1024**2

    print('Memory usage after complition: ',red_memory,'MB')

    

reduce_memory_usage(df)
df.shape
df.info()
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



def gini(y, pred):

    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)

    g = g[np.lexsort((g[:,2], -1*g[:,1]))]

    gs = g[:,0].cumsum().sum() / g[:,0].sum()

    gs -= (len(y) + 1) / 2.

    return gs / len(y)



def gini_xgb(pred, y):

    y = y.get_label()

    return 'gini', gini(y, pred) / gini(y, y)



X = df.drop('Y', axis=1)

y = df.Y



x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=4242)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(X_test_fin)



params = {

        'objective':'binary:logistic',        

        'max_depth':8,

        'learning_rate':0.07,

        'eval_metric':'auc',

        'min_child_weight':6,

        'subsample':0.8,

        'colsample_bytree':0.8,

        'seed':45,

        'reg_lambda':1.3,

        'reg_alpha':8,

        'gamma':10,

        'scale_pos_weight':1.6,

        'nthread':-1

}



watchlist = [(d_train, 'train'), (d_valid, 'valid')]

nrounds=2000  # need to change to 2000

model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 

                          feval=gini_xgb, maximize=True, verbose_eval=10)
sub = pd.DataFrame()

sub['ID'] = test['id']

sub['target'] = model.predict(d_test)

sub.to_csv('submission.csv', index=False)
sub