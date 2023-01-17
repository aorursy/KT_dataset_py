from __future__ import absolute_import, division, print_function, unicode_literals



try:

    import tensorflow.compat.v2 as tf



except Exception:

    pass



tf.enable_v2_behavior()



print(f"Tensorflow Version: {tf.__version__}")
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

%matplotlib inline

import matplotlib.pyplot as plt
import warnings as warnings

warnings.filterwarnings('ignore', category = DeprecationWarning) 

warnings.filterwarnings('ignore', category = FutureWarning) 

warnings.filterwarnings('ignore', category = UserWarning)
os.chdir('..')
train_df = pd.read_csv('input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
test_df.head()
plt.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])});

prices.hist();
#log transform the target:

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
numericColumns = []

categoricalColumns = []



for column in train_df.columns:

    if train_df[column].dtypes == int or train_df[column].dtypes==float:

        numericColumns.append(column)

    else:

        categoricalColumns.append(column)



numericColumns.remove('Id')

numericColumns.remove('SalePrice')



print( f"{len(numericColumns)} Numeric columns: {numericColumns} \n")

print( f"{len(categoricalColumns)} Categorical columns: {categoricalColumns} \n")

print( 'ID and SalePrice are seperated')
# categorical columns fillna

# firstly, to avoid loss nan during NN training

# secondly, to be able to pass to labelencoder



for column in categoricalColumns:

    train_df[column].fillna('missing', inplace = True)

    test_df[column].fillna('missing', inplace = True)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder



# labelencode string categorical column to integer categorical column

# tf.dataset cannot take in mixed data type, and thus need to change to numeric

# take care of df_test labelencoder transform, there are unique labels not in df_train



import bisect



for column in categoricalColumns:

    le = LabelEncoder()

    le.fit(train_df[column])

    train_df[column] = le.transform(train_df[column])

    le_classes = le.classes_.tolist()

    

    # to handle categorical feature only in testing data

    # handle int and string categorical columns differently

    if type(le_classes[0]) is str:

        test_df[column] = test_df[column].map(lambda s: 'other' if s not in le.classes_ else s)

        bisect.insort_left(le_classes, 'other')

        le.classes_ = le_classes

        test_df[column] = le.transform(test_df[column])

    else:

        test_df[column] = test_df[column].map(lambda s: -1 if s not in le.classes_ else s)

        bisect.insort_left(le_classes, -1)

        le.classes_ = le_classes

        test_df[column] = le.transform(test_df[column])
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size=0.2)

print(f"{len(train_df)} train examples")

print(f"{len(val_df)} validation examples")

print(f"{len(test_df)} test examples")
#numeric columns fillna, to avoid loss nan during NN training



for column in numericColumns:

    train_df[column].fillna(train_df[column].median(), inplace = True)

    val_df[column].fillna(val_df[column].median(), inplace = True)

    test_df[column].fillna(test_df[column].median(), inplace = True)
# standardscale the input data



scaler = StandardScaler()

train_df[numericColumns] = scaler.fit_transform(train_df[numericColumns])

val_df[numericColumns] = scaler.transform(val_df[numericColumns])

test_df[numericColumns] = scaler.transform(test_df[numericColumns])
def data_pipeline(dataframe, Shuffle=True, Batchsize=32):

    """This function removes the SalePrice column

    and build the data pipeline using tf.Data.Dataset.

    

    Taken from article: 

    https://towardsdatascience.com/a-succinct-tensorflow-2-0-solution-for-kaggle-house-prices-prediction-challenge-99310ad03ad0

    

    

    

    Parameters:

    -----------------------------------------------------------------------------------------------

    df(pd.DataFrame): Pandas Dataframe

    Shuffle(Boolean): If you want to shuffle data(Default=True)

    Batchsize(int): Batch Size (Default=32)

    

    """

    

    dataframe = dataframe.copy()

    labels = dataframe.pop('SalePrice')

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

    if Shuffle:

        ds = ds.shuffle(buffer_size=len(dataframe))

    ds = ds.batch(Batchsize)

    return ds
batch_size = 32

train_ds = data_pipeline(train_df, Batchsize=batch_size)

val_ds = data_pipeline(val_df, Shuffle=False, Batchsize=batch_size)

test_df['SalePrice'] = 0

test_ds = data_pipeline(test_df, Shuffle=False, Batchsize=batch_size)
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, LeakyReLU, Dropout, Concatenate

from tensorflow.keras.models import Model

from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam

from tensorflow import feature_column









def NN(dense_units, numericColumns, categoricalColumns, use_batch_norm=False,use_dropout=False):

        """Defines a neural network with dense units, the number of 

        layers is given by the lenght of dense_units.

        

        Parameters

        ------------------------------------------------------------

        input_dim(tuple): Dimension of the input.

        dense_units(list): Number of units used on each dense layer.

        use_batch_norm(Boolean): True if =False

        use_dropout=False

        """            

        

        feature_columns = []

        

        for col in numericColumns:

            col = feature_column.numeric_column(col)

            feature_columns.append(col)

        

        for col in categoricalColumns:

            col = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list(col,train_df[col].unique()))

            feature_columns.append(col)

        

        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)        

        inputs = {colname : tf.keras.layers.Input(name=colname, shape=(), dtype='float32') for colname in numericColumns}

        inputs.update({colname : tf.keras.layers.Input(name=colname, shape=(), dtype='int64') for colname in categoricalColumns})

        

        x = feature_layer(inputs)        

        

        _n_layers = len(dense_units)

        

        for i in range(_n_layers):

            

            dense_layer = Dense( 

                dense_units[i]

                , name = 'Dense_' + str(i)

                )

            

            x = dense_layer(x)



            if use_batch_norm:

                x = BatchNormalization()(x)



            x = LeakyReLU()(x)



            if use_dropout:

                x = Dropout(rate = 0.25)(x)

        

        output_model = Dense(1, name='Output')(x)

        

        model = Model(inputs, output_model)

        

        return model
initial_learning_rate = 0.001



lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(

    initial_learning_rate,

    decay_steps=10000,

    decay_rate=0.96,

    staircase=True)



optimizer = tf.optimizers.RMSprop(learning_rate=lr_schedule)



#optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)



loss_function = tf.keras.losses.mean_squared_logarithmic_error



layers = [256, 512]

learning_rate = 0.0005

model = NN(

           dense_units=layers

           ,numericColumns=numericColumns

           ,categoricalColumns=categoricalColumns

           ,use_batch_norm=True

           ,use_dropout=True

          )



model.compile(loss= loss_function,

              optimizer=optimizer,

              )



#tf.keras.utils.plot_model(model, 'Regression.png', show_shapes=False, rankdir='LR')
# train the model

history = model.fit(train_ds,

                    validation_data=val_ds,

                    epochs=10,

                    verbose=2

                   )
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch



def plot_history(history):

    hist = pd.DataFrame(history.history)

    hist['epoch'] = history.epoch

    

    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Squared Error')

    plt.plot(hist['epoch'], hist['loss'],

           label='Train Error')

    plt.plot(hist['epoch'], hist['val_loss'],

           label = 'Val Error')

    plt.legend()

    plt.show()



plot_history(history)
!ls
submission = np.expm1(model.predict(test_ds).flatten())



df = pd.DataFrame(columns = ['Id', 'SalePrice'])

df['Id'] = test_df['Id']

df['SalePrice'] = submission



df.to_csv("working/submission.csv", index=False)