import pandas as pd

import numpy as np



import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from tensorflow import feature_column

from tensorflow.keras import Sequential

from tensorflow.keras.layers import DenseFeatures, Dense, Dropout





from sklearn.model_selection import train_test_split
def missing(df):

    df_missing = pd.DataFrame(df.isna().sum().sort_values(ascending = False), columns = ['missing_count'])

    df_missing['missing_share'] = df_missing.missing_count / len(df)

    return df_missing
train_df = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test_df = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')



train_df.head(5)
len(train_df)
missing(train_df)
# Drop all train columns with any missing values

train_df = train_df.dropna(axis=1)
exclude_columns = test_df.columns[test_df.isna().any()].tolist() + ['Id', 'SalePrice']
all_numeric_columns = list(set(train_df._get_numeric_data().columns) - set(exclude_columns))

all_categorical_columns = list(set(train_df.columns) - set(all_numeric_columns) - set(exclude_columns))
all_numeric_columns
all_categorical_columns
numeric_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

categorical_features = ['SaleType', 'SaleCondition', 'LotShape', 'Neighborhood', 'ExterQual', 'ExterCond', 'HeatingQC', 'CentralAir']
features = numeric_features + categorical_features



X = train_df[features]

y = train_df.SalePrice



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size=0.1)

print(len(train_X), 'train examples')

print(len(val_X), 'validation examples')
feature_columns = []

for header in numeric_features:

    feature_columns.append(feature_column.numeric_column(header))

    

for header in categorical_features:

    categorical = feature_column.categorical_column_with_vocabulary_list(header, train_df[header].unique())

    feature_columns.append(feature_column.indicator_column(categorical))
kernel_initializer = tf.keras.initializers.GlorotNormal()

activation='relu'



model = Sequential()



# Input Layer

model.add(DenseFeatures(feature_columns))



# Hidden Layers

model.add(Dense(16, kernel_initializer=kernel_initializer, activation=activation))

#model.add(Dropout(0.2))

model.add(Dense(8, kernel_initializer=kernel_initializer, activation=activation))

#model.add(Dropout(0.2))

model.add(Dense(4, kernel_initializer=kernel_initializer, activation=activation))

# Output Layer

model.add(Dense(1, kernel_initializer=kernel_initializer, activation=activation))



model.compile(

  loss=tf.keras.losses.MeanAbsoluteError(), 

  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9), 

  metrics=[tf.keras.metrics.MeanAbsoluteError()]

)
# Convert Pandas Dataframe into tf.data dataset

def df_to_ds(X, y, shuffle=True, batch_size=32):

    ds = tf.data.Dataset.from_tensor_slices((dict(X.copy()), y))

    if shuffle:

        ds = ds.shuffle(buffer_size=len(X))

    return ds.batch(batch_size)
train_ds = df_to_ds(train_X, train_y)

val_ds = df_to_ds(val_X, val_y)
history = model.fit(train_ds,

          validation_data=val_ds,

          callbacks=[

            tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', 

                                             min_delta=200, 

                                             patience=30, 

                                             verbose=1, 

                                             restore_best_weights=True)

          ],

          epochs=300,

          verbose=0)
loss, mae = model.evaluate(val_ds)

print("Validation Loss:", loss)

print("Validation MAE:", mae)
plt.ylabel('MAE')

plt.xlabel('epoch')

plt.plot(history.history['mean_absolute_error'])

plt.plot(history.history['val_mean_absolute_error'])

plt.legend(['Train', 'Validation'])

plt.show()

test_df.dtypes
test_df[all_numeric_columns] = test_df[all_numeric_columns].astype(int)
test_X = test_df[features]

missing(test_X)
test_X[test_X.SaleType.isna()]

test_X.loc[1029, 'SaleType'] = "Oth"

missing(test_X)
test_y = pd.DataFrame(np.zeros(shape=(len(test_X),1)), columns=["SalePrice"])

test_ds = df_to_ds(test_X, test_y)
test_preds = model.predict(test_ds)

test_preds
flatten = lambda l: [item for sublist in l for item in sublist]

output = pd.DataFrame({'Id': test_df.Id,

                      'SalePrice': pd.Series(flatten(test_preds))})

output.to_csv('submission.csv', index=False)