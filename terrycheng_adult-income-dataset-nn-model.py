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

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
income = pd.read_csv("/kaggle/input/adult-income-dataset/adult.csv")

income.head()
income['income'] = income['income'].map({'<=50K':0, '>50K':1})
income['income'].value_counts()
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
init_check(income)
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

        encoded = categorical_encoding(X, categorical_cloumns=categorical_columns, encoding_method=encoding_method)

        X = X.drop(columns=categorical_columns, axis=1)

        X = pd.concat([X, encoded], axis=1)

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    

    scaler=MinMaxScaler()

    X_train= pd.DataFrame(scaler.fit_transform(X_train))

    X_test = pd.DataFrame(scaler.transform(X_test))

    

    return X_train, X_test, y_train, y_test
features = income.columns.drop('income')

X_train, X_test, y_train, y_test = data_preprocessing(df=income, features=features, 

                                                      target='income', encoding_method = 'label',

                                                      test_size=0.2, random_state=0)



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(64, activation='relu', input_shape = [X_train.shape[1]]),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(32, activation='relu'),  

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(2, activation='softmax')                     

])



optimizer = tf.keras.optimizers.RMSprop(0.001)



model.compile(optimizer=optimizer,

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
print(model.summary())
tf.keras.utils.plot_model(

    model,

    to_file='model.png',

    show_shapes=True,

    show_layer_names=True,

    rankdir='TB',

)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
plt.plot(history.history['loss'], label='train_loss')

plt.plot(history.history['val_loss'], label = 'val_loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(loc='upper right')
plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.8,0.9])

plt.legend(loc='upper right')