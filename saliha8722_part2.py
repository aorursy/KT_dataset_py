# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv(os.path.join(dirname, filename))

data.head()
data.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicides/100kpop', 'country_year', 'HDI_for_year',

       'gdp_for_year_dollars', 'gdp_per_capita_dollars', 'generation']
#gdp_for_year_dollars sütunu virgül kullanılarak string olarak kaydedilmiş, bunu numerically olacak şekilde çeviriyorum

data['gdp_for_year_dollars'] = data['gdp_for_year_dollars'].str.replace(',','').astype(int)
#numerical sütunlar-> year, suicides_no, population, suicides/100k pop, HDI for year, gdp_for_year_dollars, gdp_per_capita_dollars 

#categorical-object sütunlar-> country, sex, age, generation (kümeleme/sınıflandırma)



#country-year zaten varolan iki sütunun birleşimi olduğu için gereksiz, bu yüzden siliyorum.

del data['country_year']
data['HDI_for_year'] = data['HDI_for_year'].fillna(data['HDI_for_year'].median())

data['HDI_for_year'].isnull().any()
mask = data.dtypes == np.object

categorical_cols = data.columns[mask]
mask #categorical sütunları görebiliriz - (true)
#Kaç tane ekstra sütun oluşturulacağının belirlenmesi:

num_ohc_cols = (data[categorical_cols]

                .apply(lambda x: x.nunique())

                .sort_values(ascending=False))





# Yalnizca bir deger varsa kodlamaya gerek yoktur

small_num_ohc_cols = num_ohc_cols.loc[num_ohc_cols>1]



# one-hot sütun sayısı - kategori sayısı = 1

small_num_ohc_cols -= 1

small_num_ohc_cols.sum()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



data_ohc = data.copy()

le = LabelEncoder()

ohc = OneHotEncoder()



for col in num_ohc_cols.index:

    

    # object sütunları numerically çevirme

    dat = le.fit_transform(data_ohc[col]).astype(np.int)

    

    # orjinal sütunu dataframe'den kaldıralım

    data_ohc = data_ohc.drop(col, axis=1)



    new_dat = ohc.fit_transform(dat.reshape(-1,1))

    n_cols = new_dat.shape[1]

    col_names = ['_'.join([col, str(x)]) for x in range(n_cols)]

    new_df = pd.DataFrame(new_dat.toarray(), 

                          index=data_ohc.index, 

                          columns=col_names)

    data_ohc = pd.concat([data_ohc, new_df], axis=1)
data_ohc.shape[1] - data.shape[1]
print(data.shape[1])

data = data.drop(num_ohc_cols.index, axis=1)

print(data.shape[1]) #ilgili sütun 11den 7ye düştü
#for suicides_no



from sklearn.model_selection import train_test_split



y_col = 'suicides_no'



# splitting one-hot kodlanmamış

feature_cols = [x for x in data.columns if x != y_col]

X_data = data[feature_cols]

y_data = data[y_col]



X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 

                                                    test_size=0.3, random_state=42)

# splitting one-hot kodlanmış

feature_cols = [x for x in data_ohc.columns if x != y_col]

X_data_ohc = data_ohc[feature_cols]

y_data_ohc = data_ohc[y_col]



X_train_ohc, X_test_ohc, y_train_ohc, y_test_ohc = train_test_split(X_data_ohc, y_data_ohc, 

                                                    test_size=0.3, random_state=42)
(X_train_ohc.index == X_train.index).all()
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



LR = LinearRegression()

error_df = list()



# one-hot kodlanmamış veriler

LR = LR.fit(X_train, y_train)

y_train_pred = LR.predict(X_train)

y_test_pred = LR.predict(X_test)



error_df.append(pd.Series({'train': mean_squared_error(y_train, y_train_pred),

                           'test' : mean_squared_error(y_test,  y_test_pred)},

                           name='no enc'))

# one-hot kodlanmış veriler

LR = LR.fit(X_train_ohc, y_train_ohc)

y_train_ohc_pred = LR.predict(X_train_ohc)

y_test_ohc_pred = LR.predict(X_test_ohc)



error_df.append(pd.Series({'train': mean_squared_error(y_train_ohc, y_train_ohc_pred),

                           'test' : mean_squared_error(y_test_ohc,  y_test_ohc_pred)},

                          name='one-hot enc'))

error_df = pd.concat(error_df, axis=1)

error_df

#One-hot kodlanmış model verilere daha fazla uyacağı için, one-hot kodlanmamış modelde daha fazla error rate aldık. 
# Kopyalama uyarilariyla ayari sessize alma

pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



scalers = {'standard': StandardScaler(),

           'minmax': MinMaxScaler(),

           'maxabs': MaxAbsScaler()}



training_test_sets = {

    'not_encoded': (X_train, y_train, X_test, y_test),

    'one_hot_encoded': (X_train_ohc, y_train_ohc, X_test_ohc, y_test_ohc)}

mask = X_train.dtypes == np.float

float_columns = X_train.columns[mask]



LR = LinearRegression()



errors = {}

for encoding_label, (_X_train, _y_train, _X_test, _y_test) in training_test_sets.items():

    for scaler_label, scaler in scalers.items():

        trainingset = _X_train.copy()  # kopyalayin cunku bunu bir kereden fazla olceklemek istemiyoruz.

        testset = _X_test.copy()

        trainingset[float_columns] = scaler.fit_transform(trainingset[float_columns])

        testset[float_columns] = scaler.transform(testset[float_columns])

        LR.fit(trainingset, _y_train)

        predictions = LR.predict(testset)

        key = encoding_label + ' - ' + scaler_label + 'scaling'

        errors[key] = mean_squared_error(_y_test, predictions)



errors = pd.Series(errors)

print(errors.to_string())

print('-' * 80)

for key, error_val in errors.items():

    print(key, error_val)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set_context('talk')

sns.set_style('whitegrid')

sns.set_palette('dark')



ax = plt.axes()

ax.scatter(y_test, y_test_pred, alpha=.5)



ax.set(xlabel='Actual', 

       ylabel='Predicted',

       title='Number of Suicides Predictions using Linear Regression');
#for HDI_for_year



from sklearn.model_selection import train_test_split



y_col = 'HDI_for_year'



# splitting one-hot kodlanmamış

feature_cols = [x for x in data.columns if x != y_col]

X_data = data[feature_cols]

y_data = data[y_col]



X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 

                                                    test_size=0.3, random_state=42)

# splitting one-hot kodlanmış

feature_cols = [x for x in data_ohc.columns if x != y_col]

X_data_ohc = data_ohc[feature_cols]

y_data_ohc = data_ohc[y_col]



X_train_ohc, X_test_ohc, y_train_ohc, y_test_ohc = train_test_split(X_data_ohc, y_data_ohc, 

                                                    test_size=0.3, random_state=42)
(X_train_ohc.index == X_train.index).all() #indexlerde bir değişim olmadı
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



LR = LinearRegression()



# Hata degerleri icin depolama

error_df = list()



# one-hot kodlanmamis veriler

LR = LR.fit(X_train, y_train)

y_train_pred = LR.predict(X_train)

y_test_pred = LR.predict(X_test)



error_df.append(pd.Series({'train': mean_squared_error(y_train, y_train_pred),

                           'test' : mean_squared_error(y_test,  y_test_pred)},

                           name='no enc'))



# one-hot kodlanmis veriler

LR = LR.fit(X_train_ohc, y_train_ohc)

y_train_ohc_pred = LR.predict(X_train_ohc)

y_test_ohc_pred = LR.predict(X_test_ohc)



error_df.append(pd.Series({'train': mean_squared_error(y_train_ohc, y_train_ohc_pred),

                           'test' : mean_squared_error(y_test_ohc,  y_test_ohc_pred)},

                          name='one-hot enc'))



# Sonuclari bir araya getirelim

error_df = pd.concat(error_df, axis=1)

error_df
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



scalers = {'standard': StandardScaler(),

           'minmax': MinMaxScaler(),

           'maxabs': MaxAbsScaler()}



training_test_sets = {

    'not_encoded': (X_train, y_train, X_test, y_test),

    'one_hot_encoded': (X_train_ohc, y_train_ohc, X_test_ohc, y_test_ohc)}



# Onceden olceklendirdigimiz bir seyi olceklendirmemek icin 

# float sutunlarin listesini ve float verilerini alin 

# Orijinal verileri her seferinde ölceklememiz gerekiyor

mask = X_train.dtypes == np.float

float_columns = X_train.columns[mask]



# initialize model

LR = LinearRegression()



# tum olası kombinasyonlari tekrarlayin ve hatalari alin

errors = {}

for encoding_label, (_X_train, _y_train, _X_test, _y_test) in training_test_sets.items():

    for scaler_label, scaler in scalers.items():

        trainingset = _X_train.copy()  # kopyalayin cunku bunu bir kereden fazla olceklemek istemiyoruz.

        testset = _X_test.copy()

        trainingset[float_columns] = scaler.fit_transform(trainingset[float_columns])

        testset[float_columns] = scaler.transform(testset[float_columns])

        LR.fit(trainingset, _y_train)

        predictions = LR.predict(testset)

        key = encoding_label + ' - ' + scaler_label + 'scaling'

        errors[key] = mean_squared_error(_y_test, predictions)



errors = pd.Series(errors)

print(errors.to_string())

print('-' * 80)

for key, error_val in errors.items():

    print(key, error_val)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





sns.set_context('talk')

sns.set_style('whitegrid')

sns.set_palette('dark')



ax = plt.axes()

#  y_test, y_test_pred kullanilacak

ax.scatter(y_test, y_test_pred, alpha=.5)



ax.set(xlabel='Actual', 

       ylabel='Predicted',

       title='HDI_for_year Predictions using Linear Regression');