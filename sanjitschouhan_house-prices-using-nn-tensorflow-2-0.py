!pip install tensorflow==2.0-rc1
import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns
input_folder = "../input/house-prices-advanced-regression-techniques/"
def load_data(file):

    return pd.read_csv(input_folder+file)
raw_train_data = load_data("train.csv")

raw_test_data = load_data("test.csv")
raw_train_data.head()
na_df = raw_train_data.isna().sum()

na_df[na_df>0]
def preprocess_na(dataframe):

    df = dataframe.copy()

    df.MiscFeature = df.MiscFeature.fillna('NA')

    df.Fence = df.Fence.fillna('NA')

    df.PoolQC = df.PoolQC.fillna('NA')

    df.GarageCond = df.GarageCond.fillna('NA')

    df.GarageQual = df.GarageQual.fillna('NA')

    df.GarageFinish = df.GarageFinish.fillna('NA')

    df.GarageYrBlt = df.GarageYrBlt.fillna(0)

    df.GarageType = df.GarageType.fillna('NA')

    df.FireplaceQu = df.FireplaceQu.fillna('NA')

    df.Alley = df.Alley.fillna('NA')

    df.MasVnrType = df.MasVnrType.fillna('NA')

    df.MasVnrArea = df.MasVnrArea.fillna(0)

    df.LotFrontage = df.LotFrontage.fillna(0)

    df.BsmtQual = df.BsmtQual.fillna('NA')

    df.BsmtCond = df.BsmtCond.fillna('NA')

    df.BsmtExposure = df.BsmtExposure.fillna('NA')

    df.BsmtFinType1 = df.BsmtFinType1.fillna('NA')

    df.BsmtFinType2 = df.BsmtFinType2.fillna('NA')

    df.Electrical = df.Electrical.fillna('SBrkr')

    df.MSZoning = df.MSZoning.fillna("RL")

    df.SaleType = df.SaleType.fillna("Oth")

    df.Utilities = df.Utilities.fillna("AllPub")

    df.Exterior1st = df.Exterior1st.fillna("Other")

    df.Exterior2nd = df.Exterior2nd.fillna("Other")

    df.BsmtFinSF1 = df.BsmtFinSF1.fillna(0)

    df.BsmtFinSF2 = df.BsmtFinSF2.fillna(0)

    df.BsmtUnfSF = df.BsmtUnfSF.fillna(0)

    df.TotalBsmtSF = df.TotalBsmtSF.fillna(0)

    df.BsmtFullBath = df.BsmtFullBath.fillna(0)

    df.BsmtHalfBath = df.BsmtHalfBath.fillna(0)

    df.KitchenQual = df.KitchenQual.fillna('TA')

    df.Functional = df.Functional.fillna("Typ")

    df.GarageCars = df.GarageCars.fillna(0)

    df.GarageArea = df.GarageArea.fillna(0)

    return df
train_data = preprocess_na(raw_train_data)

test_data = preprocess_na(raw_test_data)
na_df = test_data.isna().sum()

na_df[na_df>0]
train_labels = train_data.pop('SalePrice')
describe = train_data.describe()

describe
def norm(X):

    df = X.copy()

    for column in describe:

        if df[column].dtype!='object':

            df[column] = (df[column]-describe[column]['mean'])/describe[column]['mean']

    return df
# train_data = norm(train_data)

# test_data = norm(test_data)
train_data.describe()
categorical_columns = train_data.columns[train_data.dtypes=='object']

numerical_columns = train_data.columns[train_data.dtypes!='object']

categorical_columns, numerical_columns
feature_columns = []

for column in numerical_columns:

    feature_columns.append(tf.feature_column.numeric_column(column))



for column in categorical_columns:

    vocabulary = list(set(train_data[column].unique()) | set(test_data[column].unique()))

    col = tf.feature_column.categorical_column_with_vocabulary_list(column,vocabulary)

    col_one_hot = tf.feature_column.indicator_column(col)

    feature_columns.append(col_one_hot)

feature_columns
train_data_dic = dict(train_data)

test_data_dic = dict(test_data)
ds = tf.data.Dataset.from_tensor_slices((train_data_dic, train_labels))

val = ds.take(100)

val = val.batch(32)

ds = ds.skip(100)

ds = ds.shuffle(buffer_size=len(train_data)-100)

ds = ds.batch(32)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.DenseFeatures(feature_columns))

model.add(tf.keras.layers.Dense(64, 'relu'))

model.add(tf.keras.layers.Dense(1))
optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(optimizer=optimizer, loss='mse', metrics=['mse','mae'])
early_stop = tf.keras.callbacks.EarlyStopping(patience=10)



history = model.fit(ds, epochs=200, validation_data=val, callbacks=[early_stop])
model.summary()
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.head()
plt.figure()

plt.plot(hist.epoch, hist.val_mae, label="Validation MAE")

plt.plot(hist.epoch, hist.mae, label="Train MAE")

plt.legend()
plt.figure()

plt.plot(hist.epoch, hist.val_mse, label="Validation MSE")

plt.plot(hist.epoch, hist.mse, label="Train MSE")

plt.legend()
pred = model.predict(test_data_dic)
result = pd.DataFrame(pred, columns=['SalePrice'], index=test_data.Id)
result.to_csv("submission.csv")