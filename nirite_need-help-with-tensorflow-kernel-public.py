# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install tensorflow-gpu==2.0.0-beta1
import gc

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.utils.class_weight import compute_class_weight



import tensorflow as tf

from tensorflow import feature_column

import tensorflow.keras.models

import tensorflow.keras.backend as K

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Dropout, DenseFeatures



print(tensorflow.__version__)
path = '../input/'

df_train_id = pd.read_csv(path + 'train_identity.csv')

df_train_trans = pd.read_csv(path + 'train_transaction.csv')

df_test_id = pd.read_csv(path + 'test_identity.csv')

df_test_trans = pd.read_csv(path + 'test_transaction.csv')
df_train_trans.head()
df_train_id.head()
df_train = pd.merge(df_train_trans, df_train_id, on='TransactionID', how='left')

df_test = pd.merge(df_test_trans, df_test_id, on='TransactionID', how='left')
df_train.isnull().sum().sort_values(ascending=False)[:20]
df_test.isnull().sum().sort_values(ascending=False)[:20]
miss_val_threshold = 0.25



col_to_del = []



for c in df_train.columns:

    if df_train[c].isnull().sum() > df_train.shape[0]*miss_val_threshold:

        col_to_del.append(c)



for c in df_test.columns:

    if df_train[c].isnull().sum() > df_test.shape[0]*miss_val_threshold:

        if c not in col_to_del:

            col_to_del.append(c)

        

col_to_del.append('TransactionID')

col_to_del.append('TransactionDT')
df_train.drop(columns=col_to_del, inplace = True)

df_test.drop(columns=col_to_del, inplace = True)
df_train.shape

df_test.shape
df_train.fillna(-999, inplace= True)

df_test.fillna(-999, inplace= True)
# before optimization

df_train.info()

df_test.info()
col_int16 = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',

           'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',

           'D1', 'D10', 'D15', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',

           'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30',

           'V31', 'V32', 'V33', 'V34', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59',

           'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 

           'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79',

           'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89',

           'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99',

           'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109',

           'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119',

           'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129',

           'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V279',

           'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289',

           'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299',

           'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309',

           'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319',

           'V320', 'V321']
for c in col_int16:

    df_train[c] = df_train[c].astype(np.int16)

    df_test[c] = df_test[c].astype(np.int16)
# one-hot-encoding of categorical values in train dataset

col_dummies = ['ProductCD', 'card4', 'card6', 'P_emaildomain']



for c in col_dummies:

    df_train[c] = pd.get_dummies(df_train[c])

    df_test[c] = pd.get_dummies(df_test[c])

    

# drop one-hot-encoded features     

df_train.drop(columns=col_dummies, inplace = True)  

df_test.drop(columns=col_dummies, inplace = True) 
# after optimization

df_train.info()

df_test.info()
# free memory

del df_train_id, df_train_trans, df_test_id, df_test_trans

gc.collect()
# we have extremely imbalanced data. isFraud: 0 (96,5%) / 1 (3,5%)

# we have to make sure, that we have around the same split of classes in training and validation sets:



def splitData():

    



    # 1. split training data into classes (isFraud = 1/0)

    df_train_neg = df_train.loc[df_train['isFraud'] == 0]

    df_train_pos = df_train.loc[df_train['isFraud'] == 1]

    

    # 2. split the classes into training and validation sets

    split = 0.2

    x_train_pos, x_val_pos = train_test_split(df_train_pos, test_size=split)

    x_train_neg, x_val_neg = train_test_split(df_train_neg, test_size=split)



# #     # 3. upsample minoriry class (isFalse = 1) to achieve 1:1 class distribution

#     x_train_pos = pd.concat([x_train_pos]*27)



#     # 3. downsample majority (isFraud = 0) to achieve 1:1 class distribution

    x_train_neg = x_train_neg.sample(frac=1/27) # trainings data

    x_val_neg = x_val_neg.sample(frac=1/27) # validation data (just for model fine tuning, in later versions it will be removed to use whole validation data)  



    # 4. combine and reshuffle training and validation sets

    x_train = (x_train_pos.append(x_train_neg)).sample(frac=1)

    x_val = (x_val_pos.append(x_val_neg)).sample(frac=1)



    # 5. define target values

    y_train = x_train.pop('isFraud')

    y_val = x_val.pop('isFraud')

    

    return x_train, x_val, y_train, y_val
def prepTrainData():

    

    x_train, x_val, y_train, y_val = splitData()  



    # convert pandas.DataFrames to TensorFlow.Datasets to be used in DL-models

    bs = 512    #batch size

    sh = bs*50    #shuffle buffer



    ds_train = tf.data.Dataset.from_tensor_slices((dict(x_train), y_train))

    ds_train = ds_train.shuffle(sh)

    ds_train = ds_train.batch(bs)

#     ds_train = ds_train.prefetch(buffer_size=1)          # turned this off to save memory

    

    ds_val = tf.data.Dataset.from_tensor_slices((dict(x_val), y_val))

    ds_val = ds_val.batch(bs)

#     ds_val = ds_val.prefetch(buffer_size=1)              # turned this off to save memory

    

    return ds_train, ds_val, x_train, x_val, y_train, y_val
def featureColumns(df):

    feature_columns = []

    

    for c in df.columns:

        cat = feature_column.categorical_column_with_hash_bucket(

            key=str(c),

            hash_bucket_size=df[str(c)].nunique(),

            dtype=tf.dtypes.int16)

        

        col = feature_column.embedding_column(

            categorical_column=cat,

            dimension=int(max(5, (df[str(c)].nunique())**0.5))) # embeddings dimension: max(5, sqrt of unique values)

        

        feature_columns.append(col)

        

    return feature_columns
def modelGen(feature_columns):

    model = Sequential([

        DenseFeatures(feature_columns),

        Dense(64, activation='relu'),

        Dropout(rate=0.4),

        Dense(32, activation='relu'),

        Dropout(rate=0.4),

        Dense(1, activation='sigmoid')

    ])



    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
def loopTrain(n_models):

    

    predictions = pd.DataFrame()

    

    for i in range(0, n_models):

        gc.collect()     # free memory 

        

        # prepare data

        ds_train, ds_val, x_train, x_val, y_train, y_val = prepTrainData()

        

#         # calculate classes distibution

#         clw = compute_class_weight('balanced',

#                                    np.unique(y_train),

#                                    y_train)

        

#         print('class weights: ' + str(clw))

        

        # create fearure columns

        feature_columns = featureColumns(x_train)

        

        # generat new model

        m = modelGen(feature_columns)

        m.optimizer.lr = 1e-2

        

        # training

        m.fit(ds_train,

              validation_data=ds_val,

#               class_weight = {0: clw[0], 1: clw[1]}, # balance the target class

              verbose = 1,

              epochs = 3

              )

        

        # additional metrics

        pred_class, pred_prob = metricsSklearn(x_val, y_val, m)    # additional metrics to check model prediction power

        

        # predict probability and save results

        predictions[str(i+1)] = pred_prob[:,0]

        

        print('Model ' + str(i+1) + ' finished training. \n')

        

    predictions['isFraud'] = predictions.mean(axis=1)    # mean of all predictions

                

    return predictions
# predict on validation set to get ROC-AUC and check how model predicts different classes 



def metricsSklearn(x_val, y_val, m):

    

    # convert pandas Dataframe to list

    test_data = []

    for c in x_val.columns:

        test_data.append(x_val[c])

    

    pred_class = m.predict_classes(test_data,

                                   batch_size=8192,

                                   verbose = 1)

    

    print('true classes distribution: \n'+ str(y_val.value_counts()))

    

    print('predicted classes distribution: \n' + str(pd.Series(pred_class[:,0]).value_counts()))

    

    pred_prob = m.predict(test_data,

                                 batch_size=8192,

                                 verbose = 1)

    

    print(classification_report(y_val, pred_class))

    print('ROC-AUC: ' + str(roc_auc_score(y_val, pred_prob)))

    

    return pred_class, pred_prob
n_models = 5



predictions=loopTrain(n_models)
predictions