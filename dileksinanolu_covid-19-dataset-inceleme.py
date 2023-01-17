import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


cnf = '#393e46' # confirmed - grey

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import warnings
warnings.filterwarnings('ignore')

#dataset name coronavirus-2019ncov
df = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")
df["Date"] = pd.to_datetime(df["Date"])

df_temp=pd.read_csv('/kaggle/input/covid19-global-weather-data/temperature_dataframe.csv')
df_temp["date"] = pd.to_datetime(df_temp["date"])
df_pop=pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')
df.head().T
print(df.shape[0])

# Sutun Adlari
print(df.columns.tolist())

# Veri Tipleri
print(df.dtypes)

print(df.describe())
import matplotlib.pyplot as plt
%matplotlib inline
# Matplotlib ile basit bir dagilim grafigi
ax = plt.axes()

ax.scatter(df.Deaths, df.Recovered)

# Eksenleri isimlendirme
ax.set(xlabel='Ölen Kişi',
       ylabel='İyileşen Hasta',
       title='Ölen Kişi vs İyileşen Hasta');

plt.axes().set(xlabel='Hasta Sayısı',
       ylabel='Confirmed',
       title='Dünya Genelinde Hastalar');
# Histogram
# bins = number of bar in figure
df.Confirmed.plot(kind = 'hist',bins = 20,figsize = (10,5))

plt.show()
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import plotly.offline as ply
ply.init_notebook_mode(connected=True)
import plotly.express as px
df.sort_values(by=["Confirmed"], ascending=False, inplace=True)
fig = px.pie(
    df.head(50),
    values = "Confirmed",
    names = "Country/Region",
    title = "En Yüksek Vaka Sayısına Sahip 5 Ülke"
)
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.show()
df_pop.rename(columns={'Country (or dependency)': 'country',
                             'Population (2020)' : 'population',
                             'Density (P/Km²)' : 'density',
                             'Fert. Rate' : 'fertility',
                             'Med. Age' : "age",
                             'Urban Pop %' : 'urban_percentage'}, inplace=True)
df.rename(columns={'Country/Region': 'country'}, inplace=True)
df_temp.rename(columns={'date': 'Date'}, inplace=True)
df_temp['country'] = df_temp['country'].replace('USA', 'US')
df_pop['country'] = df_pop['country'].replace('United States', 'US')
df['country'] = df['country'].replace('Mainland China', 'China')
df_pop = df_pop[["country", "population", "density", "fertility", "age", "urban_percentage"]]
df = df.merge(df_pop, on=['country'], how='left')
df_temp.drop_duplicates(subset =["Date",'country'], 
                     keep = 'first', inplace = True)
df = df.merge(df_temp, on=['Date','country'], how='left')
plt.axes().set(xlabel='Hasta Sayısı',
       ylabel='population',
       title='Dünya Genelinde Hastalar');
# Histogram
# bins = number of bar in figure
df_pop.population.plot(kind = 'hist',bins = 50,figsize = (9,4))

plt.show()

tarih=df['Date'].max()
guncel=df[df['Date']==tarih]
olum=guncel['Deaths'].sum()
iyilesme=guncel['Recovered'].sum()
vaka=guncel['Confirmed'].sum()
turkiye=guncel[guncel['country']=='Turkey']
turkiye_vaka=turkiye['Confirmed'].sum()
turkiye_olum=turkiye['Deaths'].sum()
turkiyeOlum_orani=(turkiye_olum/turkiye_vaka)*100
turkiye_iyilesme=turkiye['Recovered'].sum()
print ('Bilgilerin Son Güncellenme Tarihi: {}'.format(tarih))
print ('Türkiye Vaka: {:,.0f}'.format(turkiye_vaka))
print ('Türkiye Ölüm: {:,.0f}'.format(turkiye_olum))
print ('Türkiye İyileşme: {:,.0f}'.format(turkiye_iyilesme))
print ('Türkiye Ölüm Oranı: {:,.1f}%'.format(turkiyeOlum_orani))
print ('Toplam Ölüm: {:,.0f}'.format(olum))
print ('Toplam İyileşme: {:,.0f}'.format(iyilesme))
print ('Toplam Vaka: {:,.0f}'.format(vaka))
df['Active']=df['Confirmed']-df['Deaths']-df['Recovered']
temp = df.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='Case', value_name='Count')


fig = px.area(temp, x="Date", y="Count", color='Case',
             title='Yayılma Hızı', color_discrete_sequence = ['#21bf73', '#ff2e63', '#fe9801'])
fig.show()
df1 = pd.read_csv('../input/covid19-coronavirus/2019_nCoV_data.csv')
df1.head()
df1 = df1.astype({"Confirmed": int, "Deaths": int, "Recovered" : int})
df1 = df1.filter(["Date", "Province/State", "Country", "Last Update", "Confirmed", "Deaths", "Recovered"])
df1.head()
#Convert the date (remove time stamp)

df1['Date1'] = pd.to_datetime(df1['Date'])
df1['Date'] = df1['Date1'].dt.date
df1['Last Update1'] = pd.to_datetime(df1['Last Update'])
df1['Last Update'] = df1['Last Update1'].dt.date
df1 = df1.filter(["Date", "Province/State", "Country", "Last Update", "Confirmed", "Deaths", "Recovered"])
df1.head()
#Combine country and province to location and sum values pertaining to it

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
plt.title('Tarihe Göre Toplam Coronavirüs Vaka Sayısı')
plt.legend(df1_date.columns)
plt.xticks(rotation=75)
plt.show()
#For graph - Confirmed cases by country:

fig = px.scatter(df1_daily, x='Date', y='Confirmed', hover_data=['Province/State', 'Deaths', 'Recovered'], color='Country')
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05, xanchor='left', yanchor='bottom',
                              text='Ülkeye Göre Vakalar Sayısı',
                              font=dict(family='Calibri', size=20, color='rgb(37,37,37)'),
                              showarrow=False))
fig.update_layout(annotations = annotations)
fig.show()
df1_date = df1_date.reset_index()
df1_date
#setting index as date
df1_date['Date'] = pd.to_datetime(df1_date.Date,format='%Y-%m-%d')
df1_date.index = df1_date['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(df1_date['Confirmed'], label='Vakalar')
#setting index as date values
df1_date['Date'] = pd.to_datetime(df1_date.Date,format='%Y-%m-%d')
df1_date.index = df1_date['Date']

#sorting
data = df1_date.sort_index(ascending=True, axis=0)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df1_date)),columns=['Date', 'Confirmed'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Confirmed'][i] = data['Confirmed'][i]
new_data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data['Date'], new_data['Confirmed'], random_state = 0)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
#create features
from fastai.tabular import add_datepart
add_datepart(X_train, 'Date')
X_train.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp
X_train = X_train.filter([ "Year", "Month", "Day"])
X_train

add_datepart(X_test, 'Date')
X_test.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp
X_test = X_test.filter([ "Year", "Month", "Day"])
X_test
#implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
#make predictions and find the rmse
preds = model.predict(X_test)
rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))
rms
preds
new_data
# object (string) sutunlarini secme
mask = df1.dtypes == np.object
categorical_cols = df1.columns[mask]
# Kac tane ekstra sutun olusturulacagini belirleme
num_ohc_cols = (df1[categorical_cols]
                .apply(lambda x: x.nunique())
                .sort_values(ascending=False))


# Yalnizca bir deger varsa kodlamaya gerek yoktur
small_num_ohc_cols = num_ohc_cols.loc[num_ohc_cols>1]

# one-hot sutun satisi, kategori sayisindan bir azdir. 
small_num_ohc_cols -= 1

# Bu, orjinal sutunlarin cikarildigi varsayilan 215 sutundur.

small_num_ohc_cols.sum()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Verilerin kopyasi
data_ohc = df1.copy()

# Kodlayicilar
le = LabelEncoder()
ohc = OneHotEncoder()

for col in num_ohc_cols.index:
    
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
data_ohc.shape[1] - df1.shape[1]
print(df1.shape[1])

# dataframe'den string sutunlarin kaldirilmasi
data = df1.drop(num_ohc_cols.index, axis=1)

print(data.shape[1])
from sklearn.model_selection import train_test_split

y_col = 'Confirmed'

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
# Kopyalama uyarilariyla ayari sessize alma
pd.options.mode.chained_assignment = None
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
       # trainingset[float_columns] = scaler.fit_transform(trainingset[float_columns])
       # testset[float_columns] = scaler.transform(testset[float_columns])
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

ax.set(xlabel='Temel Doğruluk Verileri', 
       ylabel='Tahminler',
       title='Linear Regression İle Model Tahmini');
df.iloc[:, :-1].min().value_counts()
df.iloc[:, :-1].max().value_counts()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['country'] = le.fit_transform(df.country)
df['country'].sample(5)


from sklearn.preprocessing import LabelEncoder

df['province'] = pd.to_numeric(df['province'], errors='coerce')
le2= LabelEncoder()
df['province'] = le2.fit_transform(df.province)
df['province'].sample(5)
# Korelasyon değerlerini hesaplama
feature_cols = df.columns[:-1]
corr_values = df[feature_cols].corr()

# Köşegen altındaki tüm verileri boşaltarak basitleştirin
tril_index = np.tril_indices_from(corr_values)

# Kullanılmayan değerleri NaN yapın
for coord in zip(*tril_index):
    corr_values.iloc[coord[0], coord[1]] = np.NaN
    
# Verileri istifleyin ve dataframe'e dönüştürün
corr_values = (corr_values.stack().to_frame().reset_index().rename(columns={'level_0':'feature1','level_1':'feature2',0:'correlation'}))

# Sıralama için mutlak değerleri alın
corr_values['abs_correlation'] = corr_values.correlation.abs()
sns.set_context('talk')
sns.set_style('white')
sns.set_palette('dark')

ax = corr_values.abs_correlation.hist(bins=50)

ax.set(xlabel='Absolute Correlation', ylabel='Frequency');
# En yüksek korelasyon değerleri
corr_values.sort_values('correlation', ascending=False).query('abs_correlation>0.8')