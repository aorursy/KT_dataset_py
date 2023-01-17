# %matplotlib notebook



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

from sklearn.impute import SimpleImputer



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

# sub = pd.read_csv(f'../input/ieee-fraud-detection/sample_submission.csv')



# let's combine the data and work with the whole dataset

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test_identity = pd.read_csv(f'../input/ieee-fraud-detection/test_identity.csv')

test_transaction = pd.read_csv(f'../input/ieee-fraud-detection/test_transaction.csv')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
test = reduce_mem_usage(test)
train.head()
train.info()
# test.head()
# test.info()
fraud = train[train['isFraud'] == 1]
down_train = train[train['isFraud'] == 0].sample(

    n=int(len(train) / 5),

    random_state=0

)
train = down_train.append(fraud)
train = reduce_mem_usage(train)
train.info()
qual_cols = (

    ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] +

    [f'card{n}' for n in range(1, 7)] +

    [f'M{n}' for n in range(1, 10)] +

    ['DeviceType' ,'DeviceInfo'] +

    [f'id_{n}' for n in range(12, 39)]

)

print(f'Qualitative Variables: {qual_cols}')
# missing_vals = pd.DataFrame(train[train.columns].isnull().sum() * 100 / train.shape[0])

# missing_vals[missing_vals[0] > 80]
# train = train.drop(missing_vals[missing_vals[0] > 80].index, axis=1)
print(f'Duplicate Rows: {train.duplicated().sum()}')
test.columns = [i.replace('-', '_') if 'id' in i else i for i in test.columns]
int_cols = (

    train.loc[:, train.dtypes == np.int8] +

    train.loc[:, train.dtypes == np.int16] +

    train.loc[:, train.dtypes == np.int32] +

    train.loc[:, train.dtypes == np.int32]

)

int_cols = int_cols.columns
numeric_cols_train = (

     train.drop(list(qual_cols) + list(int_cols) + ['isFraud'], axis=1).columns

 )



numeric_cols_test = (

     test.drop(list(qual_cols) + list(set(int_cols) - {'isFraud'}), axis=1).columns

 )





scaler = StandardScaler()

train_numeric_norm = scaler.fit_transform(train[numeric_cols_train])

test_numeric_norm = scaler.fit_transform(test[numeric_cols_test])
test_numeric_norm = scaler.fit_transform(test[numeric_cols_train])

train_numeric_norm = scaler.fit_transform(train[numeric_cols_test])
for i, col in enumerate(numeric_cols_train):

    train[f'{col}_n'] = train_numeric_norm[:, i]

    del train[col]



for i, col in enumerate(numeric_cols_train):

    test[f'{col}_n'] = test_numeric_norm[:, i]

    del test[col]
qual_objects = train.select_dtypes(include=['object']).columns

qual_objects_test = set(train.select_dtypes(include=['object']).columns) - set('isFraud')

for col in qual_objects:

    train[col] = train[col].replace(np.nan, 'nan', regex=True)



for col in qual_objects_test:

    test[col] = test[col].replace(np.nan, 'nan', regex=True)
le = LabelEncoder()



for col in qual_objects:

    train[col] = le.fit_transform(train[col].astype(str))

    test[col] = le.fit_transform(test[col].astype(str))
for col in [f'{i}_n' for i in numeric_cols_train]:

    train[col] = train[col].replace(np.nan, 0, regex=True)

for col in [f'{i}_n' for i in numeric_cols_test]:

    test[col] = test[col].replace(np.nan, 0, regex=True)
for col in int_cols:

    train[col] = train[col].replace(np.nan, -99, regex=True)



for col in list(set(int_cols) - {'isFraud'}):

    test[col] = test[col].replace(np.nan, -99, regex=True)
for col in set(qual_cols) - set(qual_objects):

    train[col] = train[col].replace(np.nan, -99, regex=True)

    test[col] = test[col].replace(np.nan, -99, regex=True)
X = train

y = X['isFraud']

X = X.drop('isFraud', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(

    X, y, train_size=0.8, random_state=0

)



del X

del y



print(f'X Train Shape: {X_train.shape}')

print(f'X Validation Shape: {X_valid.shape}')

print(f'y Train Shape: {y_train.shape}')

print(f'y Validation Shape: {y_valid.shape}')
clf = RandomForestClassifier(random_state=0)



clf.fit(X_train, y_train)

results_valid = clf.predict(X_valid)

results = clf.predict(test)
feature_importances = pd.DataFrame(clf.feature_importances_,

                                   index = train.drop('isFraud', axis=1).columns,

                                    columns=['importance']).sort_values('importance', ascending=False)

feature_importances = feature_importances.reset_index()

feature_importances.head(10)
plt.figure(figsize=(13, 7))

sns.barplot(

    x="importance", y='index',

    data=feature_importances[0:10], label="Total"

)

plt.title("Random Forest Variable Importance")

plt.ylabel("Variable")

plt.xlabel("Importance")

plt.show()
print(classification_report(y_valid, results_valid))
results_df = pd.DataFrame()

results_df['TransactionID'] = test['TransactionID']

results_df['isFraud'] = results
results_df.to_csv('submission.csv', index=False)
rf_auc = roc_auc_score(y_valid, results_valid)

rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_valid, clf.predict_proba(X_valid)[:,1])

plt.figure(figsize=(10, 10))



# Plot Random Forest ROC

plt.plot(rf_fpr, rf_tpr, label=f'Random Forest AUC: {round(rf_auc, 3)}')





# Plot Base Rate ROC

plt.plot([0,1], [0,1],label='Base Rate')



plt.xlim([-0.005, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Graph')

plt.legend(loc="lower right")

plt.show()