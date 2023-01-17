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

from scipy import stats

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import seaborn as sbn

%matplotlib inline
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

# df_train.dtypes
# copas

# descriptive statistics summary

print(df_train['SalePrice'].describe())

#histogram

sbn.distplot(df_train['SalePrice']);

#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())

# df_train['SalePrice_old'] = df_train['SalePrice']

# df_train['SalePrice'] = np.log(df_train['SalePrice'])
#ISSUE[1] : Missing data. Really?

# comparing population with sample (population - missing data)



# select all columns that have null value, insert saleprice too for comparison

cprob_idx = df_train.isna().sum().nonzero()[0]

problem_cols = list(df_train.iloc[:,cprob_idx].columns)

problem_cols.insert(0,'SalePrice')



# get properties of columns

distdesc_df = pd.DataFrame(columns=['Column','Total','Percent','Mean','Median','Mode','Skew','Kurt','DType'])

for i, v in enumerate(problem_cols):

    _y = df_train[df_train[v].notnull()]['SalePrice']

#     print(df_train[v].dtypes)

    distdesc_df.loc[i] = [v,df_train[v].isnull().sum(),(df_train[v].isnull().sum()/df_train[v].isnull().count()),\

                          _y.mean(),_y.median(),_y.mode(),_y.skew(),_y.kurt(),df_train[v].dtypes]

# .format(df_train[v].dtypes)

# calculate distance (difference, substraction) between problem column and saleprice

distdesc_df[distdesc_df.columns[1:-1]] -= distdesc_df[distdesc_df.columns[1:-1]].iloc[0]



# show (with and) without them...

distdesc_df.sort_values(['Percent','Kurt','Skew'], ascending=False)
# distribution plot (with and) without them...

f, ax = plt.subplots(4,5, figsize=(16, 8))

for i, col in enumerate(problem_cols):

    sbn.distplot(df_train[df_train[col].notnull()]['SalePrice'], ax=ax[i%4,i%5])

    sbn.distplot(df_train['SalePrice'], ax=ax[i%4,i%5])

    ax[i%4,i%5].set_title(col)

plt.tight_layout()
# for now, drop row which have missing value but having small effects

cols_w_missing_data_selected = list(distdesc_df['Column'].loc[(distdesc_df['Percent'] < 0.1) & (distdesc_df['Percent'] != 0)])

# print("Deleting this...")

# print(cols_w_missing_data_selected)

# print(df_train[cols_w_missing_data_selected].isna().sum())

df_train_lel = df_train[cols_w_missing_data_selected].isna().any(axis=1)

df_train = df_train.drop(df_train.index[df_train_lel])

# print("...Finished.")



print("Showing sum of missing data on columns after delete within the threshold...")

print(df_train[problem_cols].isna().sum())



# # todo later: look further

# sbn.set()

# # cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# # sbn.pairplot(df_train[], size = 2.5)

# cols = ['SalePrice','GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']

# cols = ['SalePrice','BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtQual', 'BsmtCond']

# print(cols)

# sbn.pairplot(df_train[cols], size = 2.5)

# plt.show()
def preprocessing(df):

    df_mod = df.copy()

    # part1 : feature convert

    df_mod['TimeSold'] = pd.to_datetime(df_mod['MoSold'].map(str)+df_mod['YrSold'].map(str),format='%m%Y')

    df_mod['TimeSold'] = df_mod['TimeSold'].values.astype(np.int64) // 10 ** 9

    try:

        df_mod['SalePrice_old'] = df_mod['SalePrice']

        df_mod['SalePrice'] = np.log(df_mod['SalePrice'])

    except:

        pass

        

    

    # part2 : expanding MiscFeature to boolean (yes/no)

    train_miscf_df_mod = pd.get_dummies(df_mod['MiscFeature'], dummy_na=True, prefix='miscf')

    # projecting misc value to misc_df_mod

    train_miscf_df_mod = df_mod['MiscVal'].values[:,None] * train_miscf_df_mod

    

    df_mod = df_mod.join(train_miscf_df_mod)

        

    # part3 : convert all quality to ordinal types

    

    cat_rank_dict = {

        2: pd.api.types.CategoricalDtype(categories=['No', 'Yes'],ordered=True),

        51: pd.api.types.CategoricalDtype(categories=['NA', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],ordered=True),

        5: pd.api.types.CategoricalDtype(categories=['NA', 'Fa', 'TA', 'Gd', 'Ex'],ordered=True),

        6: pd.api.types.CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],ordered=True),

        7: pd.api.types.CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],ordered=True)

    }

    colname_ordinal = {

        'ExterQual': 6,

        'ExterCond': 6,

        'BsmtQual': 6,

        'BsmtCond': 6,

        'BsmtExposure': 5,

        'BsmtFinType1': 7,

        'BsmtFinType2': 7,

        'HeatingQC': 5,

        'CentralAir': 2,

        'KitchenQual': 5,

        'FireplaceQu': 6,

        'GarageQual': 6,

        'GarageCond': 6,

        'PoolQC': 5,

        'Fence': 51

    }



    for k, v in colname_ordinal.items():

        df_mod[k] = df_mod[k].astype(cat_rank_dict[v])

    # assuming NA is not 'missing value' on observation but 'poorest quality'

        try:

            df_mod[k][df_mod[k].isna()] = 'NA'

        except: 

            df_mod[k][df_mod[k].isna()] = 'No'

    #     print(df_mod_train[k].dtypes)

        df_mod[k] = df_mod[k].cat.codes



    _unknown = df_mod.select_dtypes(include='object')

    # print()

    cat_dummies = pd.get_dummies(_unknown, prefix='unkcat')

    df_mod = df_mod.join(cat_dummies)

#     print(_unknown.dtypes)

    # [ df_mod_train[col].astype('category') for col in _unknown.columns ]

    

    # last : select features

#     selected_features = list(df_mod.select_dtypes(exclude=['object','category','bool']))

    selected_features = list(df_mod.columns)

    

    # -part0

#     selected_features.remove('Id')

#     selected_features.remove('SalePrice')

    

    # -part1

    selected_features.remove('MoSold')

    selected_features.remove('YrSold')

    

    # -part2

#     selected_features.remove('MiscFeature')

    selected_features.remove('MiscVal')

    [selected_features.remove(fname) for fname in train_miscf_df_mod.columns if fname != 'miscf_Shed']

#     print(cat_dummies.columns)

    # -part3

    [selected_features.remove(fname) for fname in _unknown.columns]

#     print(selected_features)

    return df_mod[selected_features]



df_train_new = preprocessing(df_train)

df_test_new = preprocessing(df_test)

# df_train_new.isna().sum()

# df_train_new.dtypes
cat_dtype = [

        'Fence','PoolQC','FireplaceQu','GarageQual','GarageCond' 

]



f, ax = plt.subplots(1,len(cat_dtype),figsize=(24, 12), sharey=True)

# f.axis(ymin=0, ymax=800000)

for i, var in enumerate(cat_dtype):

    data = pd.concat([df_train_new['SalePrice'], df_train[var]], axis=1)

    sbn.boxplot(x=var, y="SalePrice", data=data, ax=ax[i])

# print(df_train[var].cat.codes)
# #scatter plot miscs/saleprice

# f, ax = plt.subplots(1,len(train_miscf_df.columns),figsize=(24, 4), sharey=True)

# # f.subplots_adjust(hspace=0.4, wspace=0.4, left=5, right=10)

# for i, var in enumerate(train_miscf_df.columns):

#     data = pd.concat([df_train['SalePrice_old'], df_train[var].astype(int)], axis=1)

#     data.plot.scatter(x=var, y='SalePrice_old', ylim=(0,800000), ax=ax[i]);

# plt.tight_layout()



# # #TODO drop gar2, othr, tenc

#lupa
#TODO outlier detection (z-score / absolute median maybe?)

# ...

# drop em all

# df_train.drop()
# scatterplot

win_i = 5 #knob here

sbn.set()

# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# sbn.pairplot(df_train[], size = 2.5)

selected_features = df_train_new.drop(['SalePrice'],axis=1).columns

print(len(selected_features))

# selected_features

cols = ['SalePrice', *selected_features[win_i*6:((win_i+1)*6)]]

# cols = ['SalePrice',*selected_features.columns[0:6]]

# print(win_i*6,((win_i+1)*6))

# # print(cols)

sbn.pairplot(df_train_new[cols], size = 2.5)

plt.show()
# clean data

df_train_new = df_train_new.drop(df_train_new[df_train_new['LotFrontage'] > 300].index)

df_train_new = df_train_new.drop(df_train_new[df_train_new['BsmtFinSF2'] > 1300].index)



# drop null values

print(df_train_new.shape)

df_train_new = df_train_new.dropna()

print(df_train_new.shape)

print(df_train_new.isna().sum())



# # change categorical to numbers

# cat_cols = df_train_new.select_dtypes(include=['category'])

# =============================================================================================================== #
#sampling

train_mydata = df_train_new.sample(frac=0.8,random_state=0)

valid_mydata = df_train_new.drop(train_mydata.index)

# valid_mydata.head()

# print(train_mydata.columns)

# print(train_mydata.select_dtypes(exclude=['category', 'uint8', 'float64', 'int64']).dtypes)



# get stats

train_stats = train_mydata.describe()

train_stats.pop("SalePrice")

train_stats.pop("Id")

train_stats = train_stats.transpose()

# # print(train_stats)



# # label split

train_labels = train_mydata.pop('SalePrice')

valid_labels = valid_mydata.pop('SalePrice')

train_mydata.pop('Id')

valid_mydata.pop('Id')

# print(train_mydata.isna().sum())

# normalize

def norm(x):

    _bc_stats = lambda lel: np.array(train_stats[lel][None,:]*np.ones(x.shape[0])[:,None])

    stat_mean = _bc_stats('mean')

    stat_std = _bc_stats('std')

    stat_std = np.where(stat_std == 0, 1, stat_std)

    return (x - stat_mean) / stat_std

#     print(_bc_stats('mean').shape)

#     return 0

print(train_mydata.shape)

print(train_stats.shape)



normed_train_data = norm(train_mydata)

normed_valid_data = norm(valid_mydata)

normed_train_data.head(10)
def build_model():

  model = keras.Sequential([

    keras.layers.Dense(64, activation='relu', input_shape=[len(train_mydata.keys())]),

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(1)

  ])



  optimizer = tf.keras.optimizers.RMSprop(0.001)



  model.compile(loss='mse',

                optimizer=optimizer,

                metrics=['mae', 'mse'])

  return model

model = build_model()



model.summary()
#Try the model

example_batch = normed_train_data[:10]

example_result = model.predict(example_batch)

example_result
# Display training progress by printing a single dot for each completed epoch

class PrintDot(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs):

    if epoch % 100 == 0: print('')

    print('.', end='')



EPOCHS = 1000



history = model.fit(

  normed_train_data, train_labels,

  epochs=EPOCHS, validation_split = 0.2, verbose=0,

  callbacks=[PrintDot()])
#Visualize the result from the training data

hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
def plot_history(history):

    hist = pd.DataFrame(history.history)

    hist['epoch'] = history.epoch



    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Abs Error [SalePrice]')

    plt.plot(hist['epoch'], hist['mean_absolute_error'],

           label='Train Error')

    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],

           label = 'Val Error')

    plt.ylim([0,1e+5])

    plt.legend()



    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Square Error [$SalePrice^2$]')

    plt.plot(hist['epoch'], hist['mean_squared_error'],

           label='Train Error')

    plt.plot(hist['epoch'], hist['val_mean_squared_error'],

           label = 'Val Error')

    plt.ylim([0,1e+10])

    plt.legend()

    plt.show()





plot_history(history)
#Testing the model



loss, mae, mse = model.evaluate(normed_valid_data, valid_labels, verbose=0)



print("Testing set Mean Abs Error: {:5.2f}".format(mae))
valid_predictions = model.predict(normed_valid_data).flatten()



plt.scatter(valid_labels, valid_predictions)

plt.xlabel('True Values [SalePrice]')

plt.ylabel('Predictions [SalePrice]')

plt.axis('equal')

plt.axis('square')

plt.xlim([0,plt.xlim()[1]])

plt.ylim([0,plt.ylim()[1]])

_ = plt.plot([-6e+5, 6e+5], [-6e+5, 6e+5])