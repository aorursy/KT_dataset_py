import matplotlib.pyplot as plt # side-stepping mpl backend

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import numpy as np

from sklearn import preprocessing, metrics

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, train_test_split, cross_val_score



# 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def create_col_name(base_str, start_int, end_int):

    return [base_str + str(i) for i in range(start_int, end_int+1)]



cat_cols = (['ProductCD'] + create_col_name('card', 1, 6) + ['addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] + 

            create_col_name('M', 1, 9) + ['DeviceType', 'DeviceInfo'] + create_col_name('id_', 12, 38))



id_cols = ['TransactionID', 'TransactionDT']



target = 'isFraud'



print('Categorical Columns:', cat_cols)

print('ID Columns:', id_cols)



#Creating Dataframes

dataframe_identity_train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

dataframe_transaction_train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

dataframe_transaction_test = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

dataframe_identity_test = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')



print('Identity shape:',dataframe_identity_train.shape)

print('Transaction shape:',dataframe_transaction_train.shape)

print('Test Identity shape:',dataframe_identity_test.shape)

print('Test Transaction shape:',dataframe_transaction_test.shape)
df_train = dataframe_transaction_train.merge(dataframe_identity_train, on='TransactionID', how='left')

print('df_train Shape:', df_train.shape)
df_test = dataframe_transaction_test.merge(dataframe_identity_test, on='TransactionID', how='left')

print('df_test Shape:', df_test.shape)
del dataframe_transaction_train, dataframe_identity_train

del dataframe_transaction_test, dataframe_identity_test
count_classes = pd.value_counts(df_train['isFraud'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
fig = plt.figure(figsize=(18, 6), facecolor='w')

sns.boxplot(x='isFraud', y='TransactionAmt', data=df_train)

plt.show()
fig = plt.figure(figsize=(18, 6), facecolor='w')

sns.distplot(df_train['TransactionAmt'])

plt.show()


def plot_hist(train, colname):

    _train_0 = train[train['isFraud'] == 0].reset_index(drop=False)

    _train_1 = train[train['isFraud'] == 1].reset_index(drop=False)

    

    fig = plt.figure(figsize=(12, 4), facecolor='w')

    ax = sns.kdeplot(_train_0[colname], color='b', alpha=0.4, shade=True)

    ax_2 = ax.twinx()

    sns.kdeplot(_train_1[colname], color='r', alpha=0.4, shade=True, ax=ax_2)

    ax.set_title(colname)

    plt.show()

plot_hist(df_train, 'D6')
# source https://www.kaggle.com/krishonaveen/xtreme-boost-and-feature-engineering

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
df_train = reduce_mem_usage(df_train,True)
#df_test = reduce_mem_usage(df_test,True)
import gc



gc.collect()
many_null_cols = [col for col in df_train.columns if df_train[col].isnull().sum() / df_train.shape[0] > 0.90]

print(many_null_cols)

print('Shape before drop:', df_train.shape)

df_train = df_train.drop(many_null_cols, axis = 1)

print('Shape after drop:',df_train.shape)



big_top_value_cols = [col for col in df_train.columns if df_train[col].value_counts(dropna=False, normalize=True).values[0] > 0.88]

big_top_value_cols.remove('isFraud')

df_train = df_train.drop(big_top_value_cols, axis = 1)

print(big_top_value_cols)



df_test = df_test.drop(many_null_cols, axis = 1)

df_test = df_test.drop(big_top_value_cols, axis = 1)



from sklearn  import preprocessing

for col in df_train.columns:

    if df_train[col].dtype=='object' :

        print("label encoding",col)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df_train[col].values))

        df_train[col] =lbl.transform(list(df_train[col].values))

from sklearn  import preprocessing

for col in df_test.columns:

    if df_test[col].dtype=='object' :

        print("label encoding",col)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df_test[col].values))

        df_test[col] =lbl.transform(list(df_test[col].values))
def clean_inf_nan(df):

    return df.replace([np.inf, -np.inf], np.nan)   



# Cleaning infinite values to NaN

df_train = clean_inf_nan(df_train)

df_test = clean_inf_nan(df_test) # replace all nan,inf,-inf to nan so it will be easy to replace



for i in df_train.columns:

    df_train[i].fillna(df_train[i].median(),inplace=True) # fill with median because mean may be affect by outliers.

for i in df_test.columns:

    df_test[i].fillna(df_test[i].median(),inplace=True)
# now we can split the data and train our model

X = df_train.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)

y = df_train['isFraud']

#X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)

X_test = df_test.drop(['TransactionDT', 'TransactionID'], axis=1)

#del train

test = df_test[['TransactionID']]
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.15, random_state = 0)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range = (0,1))



scaler.fit(xTrain)

xTrain = scaler.transform(xTrain)

xTest = scaler.transform(xTest)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight={0:0.85})

model.fit(xTrain, yTrain)

from sklearn.metrics import classification_report



y_pred = model.predict(xTest)

  

print(classification_report(yTest, y_pred))



print('Logistic Score:', model.score(xTest,yTest))
import csv



scaler.fit(X_test)

X_test = scaler.transform(X_test)



y_sub_predict = model.predict_proba(X_test)



csv_data = [['TransactionID', 'isFraud']]



fraud_dict = { 'fraud': 1, 'not_fraud': 0 }



for i in range(0, len(y_sub_predict)):

    csv_data.append([df_test['TransactionID'][i], y_sub_predict[i][1]])

    if y_sub_predict[i][1] >= 0.5:

        fraud_dict['fraud'] += 1

    else:

        fraud_dict['not_fraud'] += 1

print(fraud_dict)

with open('submission.csv', 'w') as csv_file:

    writer = csv.writer(csv_file)

    writer.writerows(csv_data)



csv_file.close()