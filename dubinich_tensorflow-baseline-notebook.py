# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score

from  sklearn.ensemble import RandomForestRegressor

import tensorflow as tf

plt.style.use('fivethirtyeight')

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/mldub-comp1/train_data.csv')

test = pd.read_csv('/kaggle/input/mldub-comp1/test_data.csv')

sample_sub = pd.read_csv('/kaggle/input/mldub-comp1/sample_sub.csv')
train.head()
test.head()
train.info()
numerics = ['photos', 'videos', 'comments']



for feature in numerics:

    plt.figure()

    train[feature].hist()

    plt.title(feature)

    plt.show()
# Checking minimum of each feature to safely use log transformation.

for feature in numerics:

    print('{} min value: {}'.format(feature, train[feature].min()))
# To use log the feature must be a positive number, so let's take abs and log then.

# Do not forget to apply the same transformation to test data as well!

# We are safe to use abs, because negative number of videos probably indicates parsing

# error on data preparation stage.



for feature in numerics:

    train[feature] = np.log1p(np.abs(train[feature]))

    test[feature] = np.log1p(np.abs(test[feature]))
for feature in numerics:

    plt.figure()

    train[feature].hist()

    plt.title(feature)

    plt.show()
plt.figure()

train['target_variable'].hist()

plt.title('Target distribution')

plt.show()



plt.figure()

train['target_variable'].apply(lambda x: np.log1p(x-train['target_variable'].min())).hist()

plt.title('Log-transformed target distribution')

plt.show()
train['about'].head()
train['about_len'] = train['about'].fillna('').apply(len)

test['about_len'] = test['about'].fillna('').apply(len)
train['about_len'].hist(alpha=0.4)

test['about_len'].hist(alpha=0.4)
plt.figure()

plt.scatter(train['about_len'], train['target_variable'])

plt.title('Target vs. about section length')

plt.show()
train['status'].unique()
test['status'].unique()
for df in [train, test]:

    df.loc[~df['status'].isin(['Deadpool', 'Submission', 'Confirmed']), 'status'] = 'Unknown'
# Check results

train['status'].unique()
# Here we use manual mapping, you can use other tools instead.

mapping = {'Deadpool': 0,

           'Submission': 1,

           'Confirmed': 2,

           'Unknown': -1}



train['status'] = train['status'].map(mapping)

test['status'] = test['status'].map(mapping)
selected_features = ['videos', 'comments', 'photos', 'about_len', 'status']



X_train = train[selected_features]

y_train = train['target_variable']



X_test = test[selected_features]
# Lets split our data into train and validation.

X_train['y'] = y_train

train_dataset = X_train.sample(frac=0.8,random_state=0)

validation_dataset = X_train.drop(train_dataset.index)

train_target = train_dataset.pop('y')

validation_target = validation_dataset.pop('y')
# Here we prepare our model

# Initially it is a 1 layer fully-connected regression neural network

# We are using mean-square-error as our loss



first_fully_connected_layer_shape = 8

activation = 'relu'

initial_learning_rate = 0.001

optimizer = tf.keras.optimizers.Adagrad(learning_rate = initial_learning_rate)

input_matrix_shape = train_dataset.shape[1]



model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(input_matrix_shape, input_shape=(input_matrix_shape,)),

  tf.keras.layers.Dense(first_fully_connected_layer_shape,

                        activation = activation),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(1)

])



model.compile(loss = 'mse',

              optimizer = optimizer,

              metrics = ['mae', 'mse'])

model.summary()
# Since this is regression, best practice would be to normalize our input data

# We don't need to normalize our target

def normalize(df):

    return ((df-df.mean())/df.std()).fillna(0)
EPOCHS = 10



history_train = model.fit(normalize(train_dataset), train_target,

                          epochs=EPOCHS, verbose=1)
# Predictions on validation set.

validation_predictions = model.predict(normalize(validation_dataset))

mean_absolute_error = np.mean(

    np.abs(validation_target.values - validation_predictions.flatten()))

mean_square_error = np.mean(

    np.square(validation_target.values - validation_predictions.flatten()))



print('Validation Set Metrics\nmean_absolute_error : {mae}\nmean_square_error : {mse}'.format(

    mae = mean_absolute_error, mse=mean_square_error))
train_feats = [x for x in X_train.columns if x not in ['y']]

model.fit(normalize(X_train[train_feats]),

          X_train['y'],

          epochs=EPOCHS,

          verbose=1)
preds = model.predict(X_test[train_feats])
preds.shape
submission = pd.DataFrame()

submission['id'] = test['id']

submission['target_variable'] = preds
submission.head()
submission.to_csv('submission.csv', index=False)