import numpy as np 

import pandas as pd 

import pyodbc 

import matplotlib.pyplot as plt



import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

import plotly.offline as ply

ply.init_notebook_mode(connected=True)

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")

sehir_data = pd.read_csv("../input/number-of-cases-in-the-city-covid19-turkey/number_of_cases_in_the_city.csv")
data
print(data.shape[0])

print(data.shape[1])

print(data.columns.tolist())

print(data.dtypes)
print(data['Country/Region'].value_counts())
print(data['Confirmed'].describe())
data.groupby('Country/Region').mean()
import matplotlib.pyplot as plt

%matplotlib inline
ax = plt.axes()

ax.scatter(data.Deaths, data.Recovered)



# Label the axes

ax.set(xlabel='Ölen Kişi',

       ylabel='Kurtarılan Kişi',

       title='Ölen ve Kurtarılan Kişi');
sehir_data
sehir_data.rename(columns = {"Province":"Şehir", "Number of Case":"Vaka Sayısı"},

                 inplace=True)
sehir_data.sort_values(by=["Vaka Sayısı"], ascending=False, inplace=True)
fig = px.pie(

    sehir_data.head(),

    values = "Vaka Sayısı",

    names = "Şehir",

    title = "En Yüksek Vaka Sayısına Sahip 5 Şehir"

)

fig.update_traces(textposition="inside", textinfo="percent+label")

fig.show()
sehir_temp_data = sehir_data[5:]

fig = px.pie(

    sehir_temp_data,

    values = "Vaka Sayısı",

    names = "Şehir",

    title = "Diğer Şehirlerdeki Vaka Sayısı",

    hover_data =["Vaka Sayısı"]

)

fig.update_traces(textposition="inside", textinfo="percent+label")

fig.show()
fig = px.bar(

    sehir_data[0:5],

    x = "Şehir",

    y = "Vaka Sayısı",

    title = "En Yüksek Vaka Sayısına Sahip 5 Şehir"

)

fig.update_layout(barmode="group")

fig.update_traces(

    marker_color='rgba(240, 92, 92, 0.6)',

    marker_line_color="rgba(191, 18, 18, 1)",

)

fig.show()
fig = px.bar(

    sehir_data[5:15],

    x = "Şehir",

    y = "Vaka Sayısı",

    title = "En Yüksek Vaka Sayısına Sahip [5-15] Aralığındaki Şehirler")

fig.update_layout(barmode="group")

fig.update_traces(

    marker_color='rgba(215,137,86,0.6)',

    marker_line_color="rgba(153, 83, 36, 1)",

)

fig.show()
fig = px.bar(

    sehir_data[15:],

    x = "Şehir",

    y = "Vaka Sayısı",

    title = "En Yüksek Vaka Sayısına Sahip [15-81] Aralığındaki Şehirler",

    

)

fig.update_layout(barmode="group")

fig.update_traces(marker_color='rgb(158,202,225)',

                  marker_line_color="rgb(8,48,107)",

                 )

fig.show()
df=data.filter(['Last_Update','Confirmed','Deaths','Recovered'])

df.head(75)

plt.figure(figsize=(16,8))

plt.plot(df['Confirmed'], label='Confirmed cases')
df1 = pd.read_csv('../input/covid19-coronavirus/2019_nCoV_data.csv')

df1.head()
df1 = df1.astype({"Confirmed": int, "Deaths": int, "Recovered" : int})

df1 = df1.filter(["Date", "Province/State", "Country", "Last Update", "Confirmed", "Deaths", "Recovered"])

df1.head()
df1['Date1'] = pd.to_datetime(df1['Date'])

df1['Date'] = df1['Date1'].dt.date

df1['Last Update1'] = pd.to_datetime(df1['Last Update'])

df1['Last Update'] = df1['Last Update1'].dt.date

df1 = df1.filter(["Date", "Province/State", "Country", "Last Update", "Confirmed", "Deaths", "Recovered"])

df1.head()
df1['Location'] = df1['Country'] + ', ' + df1['Province/State'].fillna('N/A')



daily = pd.DataFrame(columns=df1.columns)



for item in df1['Location'].unique():

    a = df1[df1['Location']==item].set_index('Date')

    a = a.rename_axis('Date').reset_index()

    daily = daily.append(a, sort=False, ignore_index=True)



df1_daily = daily.sort_values(['Date','Country','Province/State'])

df1_daily = df1_daily.reset_index()

df1_daily = df1_daily.filter(["Date", "Province/State", "Country", "Last Update", "Confirmed", "Deaths", "Recovered", "Location"])

df1_daily.head()
df1_date = df1_daily.filter(["Date",  "Confirmed", "Deaths", "Recovered"])

df1_date = df1_date.groupby(df1_date["Date"]).sum()

df1_date.head()
plt.figure(figsize=(11,6))

plt.plot(df1_date, marker='o')

plt.title('Total Number of Coronavirus Cases by Date')

plt.legend(df1_date.columns)

plt.xticks(rotation=75)

plt.show()
df1_date = df1_date.reset_index()

df1_date
df1_date['Date'] = pd.to_datetime(df1_date.Date,format='%Y-%m-%d')

df1_date.index = df1_date['Date']



plt.figure(figsize=(16,8))

plt.plot(df1_date['Confirmed'], label='Confirmed cases')
mask = sehir_data.dtypes == np.object

categorical_cols = sehir_data.columns[mask]
# Kac tane ekstra sutun olusturulacagini belirleme

num_ohc_cols = (sehir_data[categorical_cols]

                .apply(lambda x: x.nunique())

                .sort_values(ascending=False))





# Yalnizca bir deger varsa kodlamaya gerek yoktur

small_num_ohc_cols = num_ohc_cols.loc[num_ohc_cols>1]



# one-hot sutun satisi, kategori sayisindan bir azdir. 

small_num_ohc_cols -= 1



# Bu, orjinal sutunlarin cikarildigi varsayilan 215 sutundur.



small_num_ohc_cols.sum()
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder



# Verilerin kopyasi

data_ohc = sehir_data.copy()



# Kodlayicilar

le = LabelEncoder()

ohc = OneHotEncoder()



for col in num_ohc_cols.index:

    

    # Integer encode the string categories

    dat = le.fit_transform(data_ohc[col]).astype(np.int)

    

    # orjinal sutunu dataframeden kaldirma

    data_ohc = data_ohc.drop(col, axis=1)



    # one-hot kod verileri-- bir aralikli array dondurur

    new_dat = ohc.fit_transform(dat.reshape(-1,1))



    # Benzersiz sutun adlari olusturma

    n_cols = new_dat.shape[1]

    col_names = ['_'.join([col, str(x)]) for x in range(n_cols)]



    # Yeni dataframe olusturma

    new_df = pd.DataFrame(new_dat.toarray(), 

                          index=data_ohc.index, 

                          columns=col_names)



    # Yeni verileri dataframe'e ekleme

    data_ohc = pd.concat([data_ohc, new_df], axis=1)
# Sutun farki yukarida hesaplandigi gibidir

data_ohc.shape[1] - sehir_data.shape[1]
print(data.shape[1])



# dataframe'den string sutunlarin kaldirilmasi

data = sehir_data.drop(num_ohc_cols.index, axis=1)



print(data.shape[1])
from sklearn.model_selection import train_test_split



y_col = 'Vaka Sayısı'



# one-hot kodlanmamis verileri bolme

feature_cols = [x for x in data.columns if x != y_col]

X_data = data[feature_cols]

y_data = data[y_col]



X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 

                                                    test_size=0.3, random_state=42)

# one-hot kodlanmis verileri bolme

feature_cols = [x for x in data_ohc.columns if x != y_col]

X_data_ohc = data_ohc[feature_cols]

y_data_ohc = data_ohc[y_col]



X_train_ohc, X_test_ohc, y_train_ohc, y_test_ohc = train_test_split(X_data_ohc, y_data_ohc, 

                                                    test_size=0.3, random_state=42)
# Ayni olduklarindan emin olmak icin endeksleri karsilastirin

(X_train_ohc.index == X_train.index).all()
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



# Sonuclari bir araya getirin

error_df = pd.concat(error_df, axis=1)

error_df
# Kopyalama uyarilariyla ayari sessize alma

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

sns.set_style('ticks')

sns.set_palette('dark')



ax = plt.axes()

#  y_test, y_test_pred kullanilacak

ax.scatter(y_test, y_test_pred, alpha=.5)



ax.set(xlabel='Doğruluk Değeri', 

       ylabel='Tahmin',

       title='Linear Regression');
df1_date['Date'] = pd.to_datetime(df1_date.Date,format='%Y-%m-%d')

df1_date.index = df1_date['Date']



data = df1_date.sort_index(ascending=True, axis=0)



new_data = pd.DataFrame(index=range(0,len(df1_date)),columns=['Date', 'Confirmed'])



for i in range(0,len(data)):

    new_data['Date'][i] = data['Date'][i]

    new_data['Confirmed'][i] = data['Confirmed'][i]

new_data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_data['Date'], new_data['Confirmed'], random_state = 0)

X_train = pd.DataFrame(X_train)

X_test = pd.DataFrame(X_test)
from fastai.tabular import add_datepart

add_datepart(X_train, 'Date')

X_train.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

X_train = X_train.filter([ "Year", "Month", "Day"])

X_train



add_datepart(X_test, 'Date')

X_test.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

X_test = X_test.filter([ "Year", "Month", "Day"])

X_test
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)
preds = model.predict(X_test)

rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))

rms
preds

new_data
sehir_data.dtypes
skew = pd.DataFrame(sehir_data.skew())

skew.columns = ['skew']

skew['too_skewed'] = skew['skew'] > .75

skew