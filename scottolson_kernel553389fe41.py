# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pathlib



import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



from scipy import stats

from scipy.stats import norm, skew #for some statistics



print(f'TF version {tf.__version__}')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/*.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



# df_train.head(10)



# Any results you write to the current directory are saved as output.
df_train.head(10)

# df_train.tail()
df_train.isna().sum()
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(df_train.shape))

print("The test data size before dropping Id feature is : {} ".format(df_test.shape))



#Save the 'Id' column

df_train_ID = df_train['Id']

df_test_ID = df_test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(df_train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(df_test.shape))
fig, ax = plt.subplots()

ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()

# Drop outliers

df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(df_train['GrLivArea'], df_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
sns.distplot(df_train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])



#Check the new distribution 

sns.distplot(df_train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
ntrain = df_train.shape[0]

ntest = df_test.shape[0]

y_train = df_train.SalePrice.values

all_data = pd.concat((df_train, df_test), sort=False).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
#Correlation map to see how features are correlated with SalePrice

corrmat = df_train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data = all_data.drop(['Utilities'], axis=1)

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")



#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)



all_data.head(10)
df_train = all_data[:ntrain]

df_test = all_data[ntrain:]
def prepare_data(df):

    df = df.replace(np.nan, 0).replace(np.inf, 1e+5).replace(-np.inf, -1e+5)

    for column in df.columns:

        if df[column].dtype.name == 'object':

            df[column] = pd.Categorical(df[column]).codes

    return df



columns = df_train.columns

x_train = pd.DataFrame(df_train.to_numpy(), columns=columns)

y_train = y_train.ravel().astype(np.float64)



df_train = prepare_data(x_train)

df_test = prepare_data(df_test)



df_train.head(10)
train_stats = df_train.describe()

train_stats = train_stats.transpose()

train_stats
def normalize(df):

    return (df - train_stats['mean']) / train_stats['std']



df_train = normalize(df_train).astype(np.float64)

df_test = normalize(df_test).astype(np.float64)



print(df_train.shape)

print(df_test.shape)
df_train.head(10)
def build_model():

  model = keras.Sequential([

    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), input_shape=[len(df_train.keys())]),

    layers.Dropout(0.3),

    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),

    layers.Dropout(0.3),

    layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),

    layers.Dropout(0.3),

    layers.Dense(1)

  ])



  optimizer = tf.keras.optimizers.RMSprop(1e-4)

#   optimizer = tf.keras.optimizers.Adam(1e-4)



#   model.compile(loss='mse',

#                 optimizer=optimizer,

#                 metrics=['mae', 'mse'])



  model.compile(loss='mse',

                optimizer=optimizer,

                metrics=['mae', 'mse'])

  return model
model = build_model()
model.summary()
EPOCHS = 3000



class MyProgbarLogger(keras.callbacks.Callback):

  def on_train_begin(self, logs=None):

    self.seen = 0

    self.progbar = keras.utils.Progbar(

        target=EPOCHS,

        unit_name='epoch')



  def on_epoch_end(self, epoch, logs=None):

    self.seen += 1

    self.progbar.update(self.seen)

    

class PrintDot(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs=None):

    print('.', end='')

    

# progbar = keras.callbacks.ProgbarLogger(params={'verbose': False})

# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



history = model.fit(

  df_train, y_train,

  epochs=EPOCHS, validation_split = 0.2, verbose=0,

#   callbacks=[early_stop, MyProgbarLogger()])

    callbacks=[MyProgbarLogger()])

hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
def plot_history(history):

  hist = pd.DataFrame(history.history)

  hist['epoch'] = history.epoch



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Abs Error [SalePrice]')

  plt.plot(hist['epoch'], hist['mae'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mae'],

           label = 'Val Error')

  plt.ylim([0,10])

  plt.legend()



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Square Error [$SalePrice^2$]')

  plt.plot(hist['epoch'], hist['mse'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mse'],

           label = 'Val Error')

  plt.ylim([0,10])

  plt.legend()

  plt.show()





plot_history(history)
loss, mae, mse = model.evaluate(df_train, y_train, verbose=2)



print("Testing set Mean Abs Error: {:5.2f} SalePrice".format(mae))
test_predictions = model.predict(df_train).flatten()



plt.scatter(y_train, test_predictions)

plt.xlabel('True Values [SalePrice]')

plt.ylabel('Predictions [SalePrice]')

plt.axis('equal')

plt.axis('square')

plt.xlim([0,plt.xlim()[1]])

plt.ylim([0,plt.ylim()[1]])

_ = plt.plot([-100, 100], [-100, 100])
error = test_predictions - y_train

plt.hist(error, bins = 25)

plt.xlabel("Prediction Error [SalePrice]")

_ = plt.ylabel("Count")
y_pred = np.expm1(model.predict(df_test).flatten().ravel())

sub = pd.DataFrame()

sub["Id"] = df_test_ID

sub["SalePrice"] = y_pred

print(sub.head(20))

sub.to_csv('submission.csv', index=False)