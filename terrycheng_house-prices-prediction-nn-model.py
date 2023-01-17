import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Tensorflow builds NN

import tensorflow as tf

print('tensorflow version : ', tf.__version__)



# default libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



# for data preprocessing

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
train
print(train.shape, test.shape)
def init_check(df):

    """

    A function to make initial check for the dataset including the name, data type, 

    number of null values and number of unique varialbes for each feature.

    

    Parameter: dataset(DataFrame)

    Output : DataFrame

    """

    columns = df.columns    

    lst = []

    for feature in columns : 

        dtype = df[feature].dtypes

        num_null = df[feature].isnull().sum()

        num_unique = df[feature].nunique()

        lst.append([feature, dtype, num_null, num_unique])

    

    check_df = pd.DataFrame(lst)

    check_df.columns = ['feature','dtype','num_null','num_unique']

    check_df = check_df.sort_values(by='dtype', axis=0, ascending=True)

    

    return check_df
init_check(train)
init_check(train).query('num_null > 0')
init_check(test).query('num_null > 0')
X = train.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature','FireplaceQu'], axis=1)



X_categorical_columns = X.select_dtypes(include=['object']).columns

X_numerical_columns = X.select_dtypes(include=['float','int']).columns



# for null value in train categorical columns

X[X_categorical_columns] = X[X_categorical_columns].fillna('?')



# for null value in train numerical columns

X[X_numerical_columns] = X[X_numerical_columns].fillna(0)
X['SalePrice'].hist()
np.log(X['SalePrice']).hist()
def categorical_encoding(df, categorical_cloumns, encoding_method):

    """

    A function to encode categorical features to a one-hot numeric array (one-hot encoding) or 

    an array with value between 0 and n_classes-1 (label encoding).

    

    Parameters:

        df (pd.DataFrame) : dataset

        categorical_cloumns  (string) : list of features 

        encoding_method (string) : 'one-hot' or 'label'

    Output : pd.DataFrame

    """

    

    if encoding_method == 'label':

        print('You choose label encoding for your categorical features')

        encoder = LabelEncoder()

        encoded = df[categorical_cloumns].apply(encoder.fit_transform)

        return encoded

    

    elif encoding_method == 'one-hot':

        print('You choose one-hot encoding for your categorical features') 

        encoded = pd.DataFrame()

        for feature in categorical_cloumns:

            dummies = pd.get_dummies(df[feature], prefix=feature)

            encoded = pd.concat([encoded, dummies], axis=1)

        return encoded
def data_preprocessing(df, features, target, encoding_method, test_size, random_state):

    y = df[target]

    

    X = df[features]

    

    categorical_columns = X.select_dtypes(include=['object']).columns

    

    if len(categorical_columns) != 0 :

        encoded = categorical_encoding(df=X, categorical_cloumns=categorical_columns, encoding_method=encoding_method)

        X = X.drop(columns=categorical_columns, axis=1)

        X = pd.concat([X, encoded], axis=1)

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    

    #scaler=MinMaxScaler()

    scaler = StandardScaler()

    X_train= pd.DataFrame(scaler.fit_transform(X_train))

    X_test = pd.DataFrame(scaler.transform(X_test))

    

    return X_train, X_test, y_train, y_test
features = X.columns.drop('SalePrice')



X_train, X_valid, y_train, y_valid = data_preprocessing(df=X, features=features, 

                                                      target='SalePrice', encoding_method = 'label',

                                                      test_size=0.2, random_state=0)

y_train = np.log(y_train)

y_valid = np.log(y_valid)



print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(256, activation='relu', input_shape = [X_train.shape[1]]),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(128, activation='relu'),  

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(64, activation='relu'),  

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(1)                     

])



optimizer = tf.keras.optimizers.RMSprop(0.001)



model.compile(optimizer=optimizer,

              loss='mse',

              metrics=['mae','mse'])
print(model.summary())
tf.keras.utils.plot_model(

    model,

    to_file='model.png',

    show_shapes=True,

    show_layer_names=True,

    rankdir='TB',

)
# Display training progress by printing a single dot for each completed epoch

class PrintDot(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):

        if epoch % 100 == 0: print('')

        print('.', end='')



EPOCHS = 2000



history = model.fit(

    X_train, y_train,

    epochs=EPOCHS, 

    validation_data=(X_valid, y_valid), 

    verbose=0,

    callbacks=[PrintDot()],

)
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
def plot_history(history):

    hist = pd.DataFrame(history.history)

    hist['epoch'] = history.epoch



    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Abs Error [log(SalePrice)]')

    plt.plot(hist['epoch'], hist['mae'], label='Train Error')

    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')

    plt.ylim([0,5])

    plt.legend()



    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Square Error [$log(SalePrice)^2$]')

    plt.plot(hist['epoch'], hist['mse'], label='Train Error')

    plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')

    plt.ylim([0,5])

    plt.legend()

    plt.show()





plot_history(history)
predictions = model.predict(X_valid).flatten()

plt.scatter(y_valid, predictions)

plt.xlabel('True Values [log(SalePrice)]')

plt.ylabel('Predictions [log(SalePrice)]')

plt.axis('equal')

plt.axis('square')