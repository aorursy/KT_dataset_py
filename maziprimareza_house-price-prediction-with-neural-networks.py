# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import data

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

pd.set_option('display.max_columns', None)

df_train.head()
df_train.columns
df_train.info()
unique_id = len(set(df_train['Id'])) #set buat ngecek unique value

total = df_train.shape[0]

print("There's {} double id on the data" .format(total-unique_id))
df_train['SalePrice'].describe()
numeric_features = df_train.dtypes[df_train.dtypes != "object"].index

numeric_features = numeric_features.drop("SalePrice")

categorical_features = df_train.dtypes[df_train.dtypes == "object"].index
df_train.hist(figsize=(30,15))

plt.show()
def regplot(x, y, **kwargs):

    sns.regplot(x=x, y=y)

    x=plt.xticks(rotation=90)



f = pd.melt(df_train, id_vars=['SalePrice'], value_vars=numeric_features)

g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)

g = g.map(regplot, "value", "SalePrice")

plt.show()
def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)



f = pd.melt(df_train, id_vars=['SalePrice'], value_vars=categorical_features)

g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)

g = g.map(boxplot, "value", "SalePrice")

plt.show()
corr = df_train.corr()

fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr, vmax=.8, square=True);

plt.show()
print(corr.iloc[-1].sort_values(ascending=False).drop("SalePrice"))
df = pd.concat([df_train, df_test])

df.head()
df.shape
df_na = pd.DataFrame()

df_na["Feature"] = df.columns

missing = ((df.isnull().sum() / len(df)) * 100).values

df_na["Missing"] = missing

df_na = df_na[df_na["Feature"] != "SalePrice"]

df_na = df_na[df_na["Missing"] != 0]

df_na=df_na.sort_values(by="Missing", ascending=False)

print(df_na)
df['Alley'] = df['Alley'].fillna('None')

df['MasVnrType'] = df['MasVnrType'].fillna('None')

df['BsmtQual'] = df['BsmtQual'].fillna('None')

df['BsmtCond'] = df['BsmtCond'].fillna('None')

df['BsmtExposure'] = df['BsmtExposure'].fillna('None')

df['BsmtFinType1'] = df['BsmtFinType1'].fillna('None')

df['BsmtFinType2'] = df['BsmtFinType2'].fillna('None')

df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

df['GarageType'] = df['GarageType'].fillna('None')

df['GarageFinish'] = df['GarageFinish'].fillna('None')

df['GarageQual'] = df['GarageQual'].fillna('None')

df['GarageCond'] = df['GarageCond'].fillna('None')

df['Fence'] = df['Fence'].fillna('None')

df['MiscFeature'] = df['MiscFeature'].fillna('None')

df['PoolQC'] = df['PoolQC'].fillna('None')
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)

df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)

df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)

df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)

df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)

df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)

df['GarageCars'] = df['GarageCars'].fillna(0)

df['GarageArea'] = df['GarageArea'].fillna(0)

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
df['MSZoning'] = df['MSZoning'].fillna('RL')

df["Exterior1st"] = df["Exterior1st"].fillna('VinylSd')

df["Exterior2nd"] = df["Exterior2nd"].fillna('VinylSd')

df['Electrical'] = df['Electrical'].fillna('SBrkr')

df['KitchenQual'] = df['KitchenQual'].fillna('TA')

df['Functional'] = df['Functional'].fillna('Typ')

df['SaleType'] = df['SaleType'].fillna('WD')

df['Utilities'] = df['Utilities'].fillna('AllPub')
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
df_na = pd.DataFrame()

df_na["Feature"] = df.columns

missing = ((df.isnull().sum() / len(df)) * 100).values

df_na["Missing"] = missing

df_na = df_na[df_na["Feature"] != "SalePrice"]

df_na = df_na[df_na["Missing"] != 0]

df_na=df_na.sort_values(by="Missing", ascending=False)

print(df_na)
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
num_to_cat=["BedroomAbvGr", "BsmtFullBath", "BsmtHalfBath", "Fireplaces", "FullBath",

            "GarageCars", "HalfBath", "KitchenAbvGr", "MoSold", "MSSubClass", "OverallCond", 

            "OverallQual", "TotRmsAbvGrd", "YrSold"]



df[num_to_cat] = df[num_to_cat].apply(lambda x: x.astype("str"))
df[num_to_cat]
df = pd.get_dummies(df, drop_first=True)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

df_trans = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
df_trans.info()
train = df_trans.iloc[:df_train.shape[0]]

train = train.drop('Id', axis = 1)

test = df_trans.iloc[df_train.shape[0]:].drop("SalePrice", axis=1)



print('Size of Training Data: {}' .format(train.shape))

print('Size of Testing Data: {}' .format(test.shape))
df_trans.head()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
# bagi x dan y

X = train.drop('SalePrice', axis=1)

Y = train['SalePrice']



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=35)



print("number of training samples:",x_train.shape)

print("number of test samples:", x_test.shape)
from keras.layers import Dense

from keras.models import Sequential

from keras.losses import mean_squared_error

from keras import backend

from keras.callbacks import ModelCheckpoint

 

def rmse(y_true, y_pred):

    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
nn = Sequential()

nn.add(Dense(128, input_shape = (x_train.shape[1], ), activation = 'relu'))

nn.add(Dense(256, activation='relu'))

nn.add(Dense(1, activation = 'sigmoid'))

nn.summary()

nn.compile(optimizer='sgd', loss='mse')
J = nn.fit(x_train,y_train, epochs=200, batch_size=16, validation_split = 0.001, verbose=0)

print('MSE of the training data: {}' .format(J.history['loss'][-1]))
plt.plot(J.history['loss'][10:])
yhat = nn.predict(x_test)

yhat.shape
ytest = np.array(y_test).reshape(-1,1)

from sklearn.metrics import mean_squared_error

print('MSE of testing data: {}'.format(mean_squared_error(ytest, yhat, squared= True)))
# Function that creates our Keras model

def create_model(optimizer= 'adam' , activation= 'relu'):

    model = Sequential()

    model.add(Dense(128, input_shape=(336,), activation=activation))

    model.add(Dense(256, activation=activation))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss='mse', metrics=["accuracy"])

    return model



# Import sklearn wrapper from keras

from keras.wrappers.scikit_learn import KerasClassifier



# Create a model as a sklearn estimator

model = KerasClassifier(build_fn=create_model, epochs=6, batch_size=16, verbose = 0)
from sklearn.model_selection import RandomizedSearchCV



# Define a series of parameters

params = dict(optimizer=['sgd','adam'],batch_size=[16, 32], activation=['relu','tanh', 'sigmoid', 'softmax'])



# Create a random search cv object and fit it to the data

random_search = RandomizedSearchCV(model, param_distributions=params, cv=5)

random_search_results = random_search.fit(x_train, y_train, verbose=0)
print("Best: {} using {}".format(random_search_results.best_score_,random_search_results.best_params_))
y_pred = nn.predict(test.drop('Id',axis =1))
s = np.where(train.columns == 'SalePrice')

test.insert(23,'SalePrice', y_pred )
test.head()
result = pd.DataFrame(scaler.inverse_transform(test), columns = test.columns)
result['Id'] = round(result['Id'])

result['Id'] = result['Id'].astype(int)
res = result[['Id', 'SalePrice']]

res.to_csv('result.csv', index=False)
res