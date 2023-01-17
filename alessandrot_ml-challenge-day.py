%matplotlib notebook



import gc



# Linear Algebra

import numpy as np



# Data Processing

import pandas as pd



# Data Visualization

import seaborn as sns

import matplotlib.pyplot as plt



# Stats

from scipy import stats



# Algorithms

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_absolute_error, classification_report, roc_auc_score, roc_curve

from sklearn.preprocessing import LabelEncoder, StandardScaler



# Classifiers

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier



# Set random seed for reproducibility

np.random.seed(0)



# Stop unnecessary Seaborn warnings

import warnings

warnings.filterwarnings('ignore')

sns.set()  # Stylises graphs
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train_identity = pd.read_csv(f'../input/ieee-fraud-detection/train_identity.csv')

train_transaction = pd.read_csv(f'../input/ieee-fraud-detection/train_transaction.csv')

# test_identity = pd.read_csv(f'../input/ieee-fraud-detection/test_identity.csv')

# test_transaction = pd.read_csv(f'../input/ieee-fraud-detection/test_transaction.csv')

# sub = pd.read_csv(f'../input/ieee-fraud-detection/sample_submission.csv')



# let's combine the data and work with the whole dataset

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

# test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
train = reduce_mem_usage(train)
train.head()
train.info()
# test.head()
# test.info()
qual_cols = (

    ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] +

    [f'card{n}' for n in range(1, 7)] +

    [f'M{n}' for n in range(1, 10)] +

    ['DeviceType' ,'DeviceInfo'] +

    [f'id_{n}' for n in range(12, 39)]

)

print(f'Qualitative Variables: {qual_cols}')
missing_vals = pd.DataFrame(train[train.columns].isnull().sum() * 100 / train.shape[0])

missing_vals[missing_vals[0] > 80]
train = train.drop(missing_vals[missing_vals[0] > 80].index, axis=1)
print(f'Duplicate Rows: {train.duplicated().sum()}')
qual_cols = set(qual_cols) - set(['id_14','id_18','id_21','id_22','id_23','id_24','id_25','id_26','id_27','id_30','id_32','id_33','id_34'])
int_cols = (

    train.loc[:, train.dtypes == np.int8] +

    train.loc[:, train.dtypes == np.int16] +

    train.loc[:, train.dtypes == np.int32] +

    train.loc[:, train.dtypes == np.int32]

)

int_cols = int_cols.columns
# int_cols
# numeric_cols = (

#     train.drop(list(qual_cols) + list(int_cols), axis=1).columns

# )



# scaler = StandardScaler()

# train_numeric = scaler.fit_transform(numeric_cols)

# train = train_numeric + train[qual_cols]
#RANDOM FOREST

n_trees = 1000

max_depth = 5

split_pct_features = 0.5

clf = RandomForestClassifier(n_estimators = n_trees, max_depth = max_depth, max_features = split_pct_features, random_state=42)



clf.fit(train.drop('isFraud', axis = 1), train['isFraud'])

results = clf.predict(test_df)



# FEATURE IMPORTANCE

importance = clf.feature_importance_

importance = pd.DataFrame(importance, index = train.columns.values, columns =["importance"])

x = range(importance.shape[0])

y = importance.iloc[:,0]

yerr = importance.iloc[:,1]

plt.bar(x, y, yerr = yerr, align = "center")

plt.xlabel('features')

plt.ylabel('Feature Importance')

plt.title('Importance of Different Features')

plt.show()