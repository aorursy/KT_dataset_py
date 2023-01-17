# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/olxmobilbekas/mobilbekas.csv')
df.drop_duplicates()

df = df.drop('Penjual', axis=1)

df.shape
# drop outliers grouped by model

import numpy as np



outlier_params = {}

model_set = set()

indexes = []



for item in df['Model']:

    model_set.add(item)



for model in model_set:

    model_df = df.loc[df['Model'] == model]

    param = [model_df['Harga'].mean(),

             model_df['Harga'].std()+1]

    

    outlier_params[model] = param



for index, row in df.iterrows():

    if (np.abs((row['Harga'] - outlier_params[row['Model']][0]) / outlier_params[row['Model']][1]) >= 3):

        indexes.append(index)



print(len(indexes))

df = df.drop(index=indexes)

df['Kapasitas mesin'] = df['Kapasitas mesin'].astype('str')



print(df.shape)
# drop outliers in general



indexes = []



for item in df['Model']:

    model_set.add(item)



for model in model_set:

    model_df = df.loc[df['Model'] == model]

    param = [model_df['Harga'].mean(),

             model_df['Harga'].std()]

    

    outlier_params[model] = param



for index, row in df.iterrows():

    if (np.abs((row['Harga'] - df['Harga'].mean()) / df['Harga'].std()) >= 3):

        indexes.append(index)



print(len(indexes))

df = df.drop(index=indexes)



print(df.shape)
from sklearn.preprocessing import OneHotEncoder



# Nama Bursa Mobil is 66.2% NaN, modify it to become boolean

df['Nama Bursa Mobil'] = df['Nama Bursa Mobil'].notnull().astype('int')



df.head()
# fill NaN values by modus of corresponding model

from scipy import stats

from math import isnan

import numpy as np



model_nan_filler = {}

model_set = set()



for item in df['Model']:

    model_set.add(item)



for model in model_set:

    model_df = df.loc[df['Model'] == model]

    filler = [stats.mode(model_df['Kapasitas mesin'])[0][0],

              stats.mode(model_df['Sistem Penggerak'])[0][0],

              stats.mode(model_df['Tipe bodi'])[0][0],

              stats.mode(model_df['Tipe Penjual'])[0][0],

              stats.mode(model_df['Varian'])[0][0]]

    

    model_nan_filler[model] = filler



for index, row in df.iterrows():

    if row['Kapasitas mesin'] == 'nan' :

        df.at[index, 'Kapasitas mesin'] = model_nan_filler[row['Model']][0]

    if row['Sistem Penggerak'] is np.nan:

        df.at[index, 'Sistem Penggerak'] = model_nan_filler[row['Model']][1]

    if row['Tipe bodi'] is np.nan:

        df.at[index, 'Tipe bodi'] = model_nan_filler[row['Model']][2]

    if row['Tipe Penjual'] is np.nan:

        df.at[index, 'Tipe Penjual'] = model_nan_filler[row['Model']][3]

    if row['Varian'] is np.nan:

        df.at[index, 'Varian'] = model_nan_filler[row['Model']][4]



df['Kapasitas mesin'] = df['Kapasitas mesin'].astype('str')



# drop drows that has NaN as its most frequent value

df = df.dropna()

print(df.shape)

df.head(100)
# drop merek that is too few in data to be predicted

df = df[df.Merek != 'Aston Martin']

df = df[df.Merek != 'Bentley']

df = df[df.Merek != 'Cadillac']

df = df[df.Merek != 'Chery']

df = df[df.Merek != 'Chrysler']

df = df[df.Merek != 'Citroen']

df = df[df.Merek != 'DFSK (Dongfeng Sokon)']

df = df[df.Merek != 'Daewoo']

df = df[df.Merek != 'Dodge']

df = df[df.Merek != 'Ferrari']

df = df[df.Merek != 'Fiat']

df = df[df.Merek != 'Foton']

df = df[df.Merek != 'Geely']

df = df[df.Merek != 'Holden']

df = df[df.Merek != 'Hummer']

df = df[df.Merek != 'Infiniti']

df = df[df.Merek != 'Jaguar']

df = df[df.Merek != 'Klasik dan Antik']

df = df[df.Merek != 'Lain-lain']

df = df[df.Merek != 'Lamborghini']

df = df[df.Merek != 'Maserati']

df = df[df.Merek != 'Mobil CBU']

df = df[df.Merek != 'Opel']

df = df[df.Merek != 'Renault']

df = df[df.Merek != 'Roll-Royce']

df = df[df.Merek != 'Smart']

df = df[df.Merek != 'Ssang Yong']

df = df[df.Merek != 'Subaru']

df = df[df.Merek != 'Tata']

df = df[df.Merek != 'Volvo']
# drop data with value 0 grouped by model

indexes = []



for index, row in df.iterrows():

    if row['Kapasitas mesin'][0] != '>' and row['Kapasitas mesin'][0] != '<':

        indexes.append(index)

        

print(indexes)

df = df.drop(indexes)



print(df.shape)
# turn jarak tempuh and kapasitas mesin to numerical data

for index, row in df.iterrows():

    df.at[index, 'Jarak tempuh'] = row['Jarak tempuh'].split('-')[-1].replace(' km', '').replace('<', '').replace('>', '')

    

    df.at[index, 'Kapasitas mesin'] = row['Kapasitas mesin'].split('-')[-1][1:].replace(' cc', '')



df.head()
# drop data with value 0 grouped by model

indexes = []



for index, row in df.iterrows():

    if row['Sistem Penggerak'] == 0 or row['Tipe bodi'] == 0 or row['Tipe Penjual'] == 0:

        indexes.append(index)

        

print(indexes)

df = df.drop(indexes)



print(df.shape)
# seperate lokasi

from sklearn.preprocessing import minmax_scale

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



locations = []

for index, row in df.iterrows():

    locations.append(row['Lokasi'].replace(' ','').split(','))

locations = pd.DataFrame(locations)
# preprocess categorical data

from numpy import array

from sklearn import preprocessing



encoded_df = df



one_hot = pd.get_dummies(locations[0])

encoded_df = encoded_df.drop('Lokasi',axis = 1)

encoded_df = encoded_df.join(one_hot.add_prefix('Kecamatan_'))



one_hot = pd.get_dummies(locations[1])

encoded_df = encoded_df.join(one_hot.add_prefix('Kabupaten_'))



one_hot = pd.get_dummies(locations[2])

encoded_df = encoded_df.join(one_hot.add_prefix('Provinsi'))



one_hot = pd.get_dummies(df['Merek'])

encoded_df = encoded_df.drop('Merek',axis = 1)

encoded_df = encoded_df.join(one_hot.add_prefix('Merek'))



one_hot = pd.get_dummies(encoded_df['Varian'])

encoded_df = encoded_df.drop('Varian',axis = 1)

encoded_df = encoded_df.join(one_hot.add_prefix('Varian_'))



one_hot = pd.get_dummies(encoded_df['Model'])

encoded_df = encoded_df.drop('Model',axis = 1)

encoded_df = encoded_df.join(one_hot.add_prefix('Model_'))



encoded_df['Jarak tempuh'] = minmax_scale(encoded_df['Jarak tempuh'])



one_hot = pd.get_dummies(encoded_df['Tipe bahan bakar'])

encoded_df = encoded_df.drop('Tipe bahan bakar',axis = 1)

encoded_df = encoded_df.join(one_hot.add_prefix('TipeBahanBakar_'))



one_hot = pd.get_dummies(encoded_df['Warna'])

encoded_df = encoded_df.drop('Warna',axis = 1)

encoded_df = encoded_df.join(one_hot.add_prefix('Warna_'))



one_hot = pd.get_dummies(encoded_df['Transmisi'])

encoded_df = encoded_df.drop('Transmisi',axis = 1)

encoded_df = encoded_df.join(one_hot.add_prefix('Transmisi_'))



one_hot = pd.get_dummies(encoded_df['Sistem Penggerak'])

encoded_df = encoded_df.drop('Sistem Penggerak',axis = 1)

encoded_df = encoded_df.join(one_hot.add_prefix('SistemPenggerak_'))



one_hot = pd.get_dummies(encoded_df['Tipe bodi'])

encoded_df = encoded_df.drop('Tipe bodi',axis = 1)

encoded_df = encoded_df.join(one_hot.add_prefix('TipeBodi_'))



encoded_df['Kapasitas mesin'] = minmax_scale(encoded_df['Kapasitas mesin'])



encoded_df['Tipe Penjual'] = le.fit_transform(encoded_df['Tipe Penjual'])



pd.set_option('display.max_columns', 500)

encoded_df['Tahun'].replace('<1986', 1986, inplace=True)



from sklearn.preprocessing import minmax_scale

encoded_df['Tahun'] = minmax_scale(encoded_df['Tahun'])



encoded_df
# normalize harga so the error is managable

from sklearn.preprocessing import minmax_scale



encoded_df['Harga'] = encoded_df['Harga']/1000000

encoded_df = encoded_df.dropna()

encoded_df.head()
# split data and label; train and test

from sklearn.model_selection import train_test_split



X = encoded_df.iloc[:,1:]

y = encoded_df.iloc[:,:1]



## dimention reduction with tSNE

# from sklearn.manifold import TSNE

# X = TSNE(n_components=3).fit_transform(X)

# X.shape



## dimention reduction with isomap

# from sklearn.manifold import Isomap

# isomap = Isomap(n_components=200, n_jobs = 4, n_neighbors = 5)

# X = isomap.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# initial model without much preprocessing



## linear regression

from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



y_pred = model.predict(X_train)



print("R2 score : %.2f" % r2_score(y_train, y_pred))

print("Data train Mean absolute error: %2f" % mean_absolute_error(y_train, y_pred))

print("Data train Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))

print("Ratio of error with label (mean): {0:.2f}%".format(mean_absolute_error(y_train, y_pred)/encoded_df['Harga'].mean()*100))
y_pred = model.predict(X_test)



print("R2 score : %.2f" % r2_score(y_test, y_pred))

print("Data train Mean absolute error: %2f" % mean_absolute_error(y_test, y_pred))

print("Data train Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

print("Ratio of error with label (mean): {0:.2f}%".format(mean_absolute_error(y_test, y_pred)/encoded_df['Harga'].mean()*100))
## regression with Ridge

from sklearn.linear_model import Ridge



model = Ridge(alpha=1.0, fit_intercept=True, solver='sag').fit(np.ascontiguousarray(X_train, dtype=np.float32), y_train)
y_pred = model.predict(X_train)



print("Data train R2 score : %.2f" % r2_score(y_train, y_pred))

print("Data train Mean Absolute Error: %.2f" % mean_absolute_error(y_train, y_pred))

print("Data train Mean Squared Error: %.2f" % mean_squared_error(y_train, y_pred))

print("Ratio of error with label (mean): {0:.2f}%".format(mean_absolute_error(y_train, y_pred)/encoded_df['Harga'].mean()*100))
y_pred = model.predict(X_test)



print("Data test R2 score : %.2f" % r2_score(y_test, y_pred))

print("Data test Mean Absolute Error: %2f mio" % mean_absolute_error(y_test, y_pred))

print("Data test Mean Squared Error: %.2f mio" % mean_squared_error(y_test, y_pred))

print("Ratio of error with label (mean): {0:.2f}%".format(mean_absolute_error(y_test, y_pred)/encoded_df['Harga'].mean()*100))
## polynomial regression with Lasso

from sklearn.linear_model import Lasso



model = Lasso(alpha=1.0, fit_intercept=True).fit(X_train, y_train)
y_pred = model.predict(X_train)



print("Data train R2 score : %.2f" % r2_score(y_train, y_pred))

print("Data train Mean Absolute Error: %.2f mio" % mean_absolute_error(y_train, y_pred))

print("Data train Mean Squared Error: %.2f mio" % mean_squared_error(y_train, y_pred))

print("Ratio of error with label (mean): {0:.2f}%".format(mean_absolute_error(y_train, y_pred)/encoded_df['Harga'].mean()*100))
y_pred = model.predict(X_test)



print("Data test R2 score : %.2f" % r2_score(y_test, y_pred))

print("Data test Mean Absolute Error: %2f mio" % mean_absolute_error(y_test, y_pred))

print("Data test Mean Squared Error: %.2f mio" % mean_squared_error(y_test, y_pred))

print("Ratio of error with label (mean): {0:.2f}%".format(mean_absolute_error(y_test, y_pred)/encoded_df['Harga'].mean()*100))
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel



model = GaussianProcessRegressor().fit(X_train, y_train)
y_pred = model.predict(X_train)



print("Data train R2 score : %.2f" % r2_score(y_train, y_pred))

print("Data train Mean Absolute Error: %2f mio" % mean_absolute_error(y_train, y_pred))

print("Data train Mean Squared Error: %.2f mio" % mean_squared_error(y_train, y_pred))

print("Ratio of error with label (mean): {0:.2f}%".format(mean_absolute_error(y_train, y_pred)/encoded_df['Harga'].mean()*100))
y_pred = model.predict(X_test)



print("Data test R2 score : %.2f" % r2_score(y_test, y_pred))

print("Data test Mean Absolute Error: %2f mio" % mean_absolute_error(y_test, y_pred))

print("Data test Mean Squared Error: %.2f mio" % mean_squared_error(y_test, y_pred))

print("Ratio of error with label (mean): {0:.2f}%".format(mean_absolute_error(y_test, y_pred)/encoded_df['Harga'].mean()*100))
import pandas

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
# custom R2-score metrics for keras backend

from keras import backend as K



def coeff_determination(y_true, y_pred):

    SS_res =  K.sum(K.square( y_true-y_pred ))

    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )

    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



# define base model

def baseline_model(dim_shape):

    # create model

    model = Sequential()

    model.add(Dense(128, input_dim=dim_shape, kernel_initializer='normal', activation='relu'))

    

    model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    

    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile model

    model.compile(loss='mae', 

                  optimizer='adam', 

                  metrics=[coeff_determination, 'mse'])

    return model



# k-fold (ultimately not used)

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold



# define 10-fold cross validation test harness

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

kfold_stratified = StratifiedKFold(n_splits=10, shuffle=True, random_state=24)



nn_train_r2_scores = []

nn_test_r2_scores = []



nn_train_mae_scores = []

nn_test_mae_scores = []



nn_train_mse_scores = []

nn_test_mse_scores = []



model_nn = baseline_model(X.shape[1])



# fit the model

model_nn.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)



# evaluate the model

loss, r2, mse = model_nn.evaluate(X_train, y_train, verbose=0)

nn_train_r2_scores.append(r2)

nn_train_mae_scores.append(loss)

nn_train_mse_scores.append(mse)



loss, r2, mse = model_nn.evaluate(X_test, y_test, verbose=0)

nn_test_r2_scores.append(r2)

nn_test_mae_scores.append(loss)

nn_test_mse_scores.append(mse)
y_pred = model.predict(X_train)



print("Data train R2 score : {0:.2f}".format(np.mean(nn_train_r2_scores),np.std(nn_train_r2_scores)))

print("Training set Mean Abs Error: {0:.2f} mio".format(np.mean(nn_train_mae_scores)))

print("Training set Mean Squared Error: {0:.2f} mio".format(np.mean(nn_train_mse_scores)))

print("Ratio of error with label (mean): {0:.2f}%".format(np.mean(nn_train_mae_scores)/encoded_df['Harga'].mean()*100))
y_pred = model.predict(X_test)



print("Data test R2 score : {0:.2f}".format(np.mean(nn_test_r2_scores),np.std(nn_test_r2_scores)))

print("Testing set Mean Abs Error: {0:.2f} mio".format(np.mean(nn_test_mae_scores)))

print("Testing set Mean Squared Error: {0:.2f} mio".format(np.mean(nn_test_mse_scores)))

print("Ratio of error with label (mean): {0:.2f}%".format(np.mean(nn_test_mae_scores)/encoded_df['Harga'].mean()*100))