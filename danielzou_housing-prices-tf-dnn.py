# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import functools ## used in numerical data processing standarization



from matplotlib import pyplot as plt

###

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



### Deep Learning Technology May perform better on Larger Data set. This is just a demonstration of DNN workflow:

## Loading Data, feature engineering, traning and plotting costs.
! head -n 100 {'/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt'}
train_file_path = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'

test_file_path = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'
!head {train_file_path}

!wc -l {train_file_path}
!head {test_file_path}

!wc -l {test_file_path}
## Load csv

column_names = "Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition,SalePrice"

column_names = column_names.split(',')

feature_names = column_names[:-1]

label_name = column_names[-1]



def get_train_dataset(file_path, **kwargs):

    dataset = tf.data.experimental.make_csv_dataset(

          file_path,

          batch_size=5, # Artificially small to make examples easier to show.

          column_names=column_names,

          label_name=label_name,

          na_value="NA",

          num_epochs=1,

          ignore_errors=True, 

          **kwargs)

    return dataset

def get_test_dataset(file_path, **kwargs):

    dataset = tf.data.experimental.make_csv_dataset(

          file_path,

          batch_size=5, # Artificially small to make examples easier to show.

          column_names=feature_names,

          na_value="NA",

          num_epochs=1,

          ignore_errors=True, 

          **kwargs)

    return dataset

raw_train_data = get_train_dataset(train_file_path)

raw_test_data = get_test_dataset(test_file_path)
def show_batch(dataset):

    for batch, label in dataset.take(1):

        for key, value in batch.items():

            print("{:20s}: {}".format(key,value.numpy()))

def show_test_batch(dataset):

    for batch in dataset.take(1):

        for key, value in batch.items():

            print("{:20s}: {}".format(key,value.numpy()))

show_batch(raw_train_data)
train_df = pd.read_csv(train_file_path,index_col='Id')

cols = train_df.columns
print(cols)
feature_columns = []

## Read Data Specifications carefully

NUMERIC_FEATURES = [ 'LotFrontage', 'LotArea','MasVnrArea','BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',  '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces', 'GarageCars', 'GarageArea', 

        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea','MiscVal']



class PackTrainNumericFeatures(object):

    def __init__(self, names):

        self.names = names



    def __call__(self, features, labels):

        numeric_features = [features.pop(name) for name in self.names]

        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]  ## Although Some features are integers, I don't want to be careful here.

        numeric_features = tf.stack(numeric_features, axis=-1)

        features['numeric'] = numeric_features

        return features, labels

class PackTestNumericFeatures(object):

    def __init__(self, names):

        self.names = names



    def __call__(self, features):

        numeric_features = [features.pop(name) for name in self.names]

        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]

        numeric_features = tf.stack(numeric_features, axis=-1)

        features['numeric'] = numeric_features

        return features

packed_num_train_data = raw_train_data.map(

    PackTrainNumericFeatures(NUMERIC_FEATURES))



packed_num_test_data = raw_test_data.map(

    PackTestNumericFeatures(NUMERIC_FEATURES))
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()

MEAN = np.array(desc.T['mean'])

STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):

    # Center the data

    return (data-mean)/std

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)



numeric_column = tf.feature_column.numeric_column('numeric',default_value=0, normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])

feature_columns.append(numeric_column)
# bucketized cols

BUCKET_FEATURES =['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']



## Explored the boundaries by: train_df.YearBuilt.describe()

YearBuilt = tf.feature_column.numeric_column("YearBuilt")

YearBuilt_buckets = tf.feature_column.bucketized_column(YearBuilt, boundaries=[*range(1880,2001,10)])

feature_columns.append(YearBuilt_buckets)



YearRemodAdd = tf.feature_column.numeric_column("YearRemodAdd")

YearRemodAdd_buckets = tf.feature_column.bucketized_column(YearRemodAdd, boundaries=[*range(1960,2001,10)])

feature_columns.append(YearRemodAdd_buckets)



GarageYrBlt = tf.feature_column.numeric_column("GarageYrBlt")

GarageYrBlt_buckets = tf.feature_column.bucketized_column(GarageYrBlt, boundaries=[*range(1910,2001,10)])

feature_columns.append(GarageYrBlt_buckets)



YrSold = tf.feature_column.numeric_column("YrSold")

YrSold_buckets = tf.feature_column.bucketized_column(YrSold, boundaries=[*range(2007,2010,1)])

feature_columns.append(YrSold_buckets)
## Indicator Categorical

col_left = [col for col in cols if ((col not in NUMERIC_FEATURES) and (col not in BUCKET_FEATURES)) and col != 'SalePrice' and col !='Id']

CATEGORIES = {key: [i for i in list(train_df[key].unique()) if str(i)!= 'nan'] for key in col_left}



for feature, vocab in CATEGORIES.items():

    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(

        key=feature, vocabulary_list=vocab)

    feature_columns.append(tf.feature_column.indicator_column(cat_col))

preprocessing_layer = tf.keras.layers.DenseFeatures(feature_columns)
train_ds = packed_num_train_data.unbatch()

N_VALIDATION = 256

N_TRAIN = len(train_df) - N_VALIDATION

BUFFER_SIZE = N_TRAIN

BATCH_SIZE = 64

STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE



validate_ds = train_ds.take(N_VALIDATION).cache()

train_ds = train_ds.skip(N_VALIDATION).take(N_TRAIN).cache()



validate_ds = validate_ds.batch(BATCH_SIZE)

train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(

  0.1,

  decay_steps=STEPS_PER_EPOCH*1000,

  decay_rate=1,

  staircase=False)



step = np.linspace(0,100000)

lr = lr_schedule(step)

plt.figure(figsize = (8,6))

plt.plot(step/STEPS_PER_EPOCH, lr)

plt.ylim([0,max(plt.ylim())])

plt.xlabel('Epoch')

_ = plt.ylabel('Learning Rate')



def get_optimizer():

    return tf.keras.optimizers.Adam(lr_schedule)



def get_callbacks(name):

    return [

    tf.keras.callbacks.EarlyStopping(monitor='val_MSLE', patience=200), ## monitor on the validation loss

    tf.keras.callbacks.EarlyStopping(monitor='MSLE', patience=200), ## monitor on the training loss

  ]



def compile_and_fit(model, name, optimizer=None, max_epochs=10000):

    if optimizer is None:

        optimizer = get_optimizer()

    model.compile(optimizer=optimizer,

                loss=tf.keras.losses.MeanSquaredLogarithmicError(),

                metrics=[

                  tf.keras.losses.MeanSquaredLogarithmicError(name='MSLE'),

                  tf.keras.metrics.RootMeanSquaredError(name='RMSE')])

    history = model.fit(

        train_ds,

        steps_per_epoch = STEPS_PER_EPOCH,

        epochs=max_epochs,

        validation_data=validate_ds,

        callbacks=get_callbacks(name),

        verbose=1)

    

    model.summary()

    return history

def plotHistory(history):

    plt.clf()

    # summarize history for accuracy

    plt.plot(history.history['MSLE'])

    plt.plot(history.history['val_MSLE'])

    plt.title('model MSLE')

    plt.ylabel('MSLE')

    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()



model_histories = {}
tf.keras.backend.clear_session()

## Define DNN1

DNN1 = tf.keras.Sequential([

    preprocessing_layer,

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001)),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001)),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1),

])



model_histories['DNN1'] = compile_and_fit(DNN1, 'models/DNN1')

plotHistory(model_histories['DNN1'])

### Multiple Trials skipped for hyperparameter tuning
## Train using best model and all data



### all data

final_ds = packed_num_train_data.unbatch()

N_TRAIN = len(train_df)

BUFFER_SIZE = N_TRAIN

BATCH_SIZE = 64

STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

final_ds = final_ds.take(N_TRAIN).cache()

final_ds = final_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)



## final compile and fit

def get_callbacks_final(name):

    return [

    tf.keras.callbacks.EarlyStopping(monitor='MSLE', patience=300), ## monitor on the MSLE

  ]



def compile_and_fit_final(model, name, optimizer=None, max_epochs=10000):

    if optimizer is None:

        optimizer = get_optimizer()

    model.compile(optimizer=optimizer,

                loss=tf.keras.losses.MeanSquaredLogarithmicError(),

                metrics=[

                  tf.keras.losses.MeanSquaredLogarithmicError(name='MSLE'),

                  tf.keras.metrics.RootMeanSquaredError(name='RMSE')])

    history = model.fit(

        train_ds,

        steps_per_epoch = STEPS_PER_EPOCH,

        epochs=max_epochs,

        callbacks=get_callbacks_final(name),

        verbose=1)

    

    model.summary()

    return history

tf.keras.backend.clear_session()

## Define DNN3

DNN3 = tf.keras.Sequential([

    preprocessing_layer,

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(80, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0005)),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(80, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0005)),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(80, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0005)),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(80, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0005)),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(80, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0005)),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1),

    ])



model_histories['DNN_final'] = compile_and_fit_final(DNN3, 'models/DNN_final')


## Begin Submission Process

packed_num_test_data = packed_num_test_data.unbatch().batch(1460)


def show_predict(dataset):

    for batch in dataset.take(-1):

        Ids = batch['Id'].numpy()

        preds = DNN3.predict_on_batch(batch).numpy().reshape(-1)

        df = pd.DataFrame({'Id':Ids, 'SalePrice':preds})

    return df

df=show_predict(packed_num_test_data)
df = df.set_index('Id').sort_index()
df.to_csv("submission.csv")