import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import callbacks
from keras import optimizers
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/used-cars-database/autos.csv', sep=',', header=0, encoding='cp1252')
df.shape
df.head(10)
df.dtypes
df.describe()
nulls_summary = pd.DataFrame(df.isnull().any(), columns=['Nulls'])
nulls_summary['Num_of_nulls [qty]'] = pd.DataFrame(df.isnull().sum())
nulls_summary['Num_of_nulls [%]'] = round((df.isnull().mean()*100),2) 
nulls_summary
df.skew()
df['price'].hist()
mean = df['price'].mean()
df['CP'] = df['price'].apply(lambda x: 'small' if x < 0.01*mean else ('large' if x > 500 * mean else 'ok'))
#usuwam dane które posiadają skrajne wartości ceny
df = df[df['CP'] == 'ok']
x = df['price']
num_bins = 5000

fig = plt.figure(figsize=(17,10))
plt.xlim([0.0,60000])
plt.hist(x, num_bins)
plt.show()
#histogram prezentuje, iż najwiecej jest pozycji o cenie mniejszej niż 1000.
df['kilometer'].mean()
df['kilometer'].min()
mean = df['kilometer'].mean()
df['C2'] = df['kilometer'].apply(lambda x: 'small' if x < 0.01*mean else ('large' if x > 500 * mean else 'ok'))
#pozbywam się danych które są wynikiem błędów (widzę kręcenie licznika ale bez przesady)
df.groupby(by='C2').count()
x = df['kilometer']
num_bins = 50

fig = plt.figure(figsize=(17,10))
plt.xlim([0.0,600000])
plt.ylim([0.0,200000])

plt.hist(x, num_bins)
plt.show()
#Histogram pokazuje, iż najwiecej aut jest z przebiegiem około 140k.
#Niestety wykres bardzo odbiega od rozkłądu normalnego
#Możliwe, że sprzedawcy sprzedają tylko i wyłącznie auta do określonego przebiegu.
df[['seller','price']].groupby(by=['seller']).count()
#kolumna posiada tylko dwie katygorie. Dodatwkowo jedna kategoria stanowi promil zbioru danych.
df[df['seller']=='gewerblich']
#kategoria posiada za małą ilość danych aby być brana pod uwagę.
df = df[df['seller']=='privat']
#zmieniam zmienna katygoryczna na numeryczne których użyję do budowy modelu.
from sklearn.preprocessing import LabelEncoder
df = df.drop(columns=['seller'])
df[['offerType','price']].groupby(by='offerType').count()
#podobnie jak w typie sprzedawcy, kolumna zawiera zbyt małą ilość odmiennych rekordów.
#Usuwamy kolumne.
df = df[df['offerType']=='Angebot']
df = df.drop(columns=['offerType'])
df.groupby(by='notRepairedDamage')['price'].count()
df[df['notRepairedDamage'].isnull()].price.count()
df[df['notRepairedDamage'].isnull()].price.count()
#dane mimo braków nadają się do stworzenia modelu.
#można stworzyć model wykorzystując tylko dane które są (jest ich 80%).
#prawdopodobnie ta kolumna będzie miała duży wpływ na cene.
df['notRepairedDamage'] = df['notRepairedDamage'].fillna('NAN')
lb_make = LabelEncoder()
df['notRepairedDamage'] = lb_make.fit_transform(df["notRepairedDamage"])
df.groupby(by='gearbox')['price'].count()
df[df['gearbox'].isnull()].price.count()
#kolumna posiada braki ale można swobodnie jej użyć do nauki modelu.
df['gearbox'] = df['gearbox'].fillna('NAN')
lb_make = LabelEncoder()
df['gearbox'] = lb_make.fit_transform(df['gearbox'])
df['powerPS']
x = df['powerPS']
num_bins = 5000

fig = plt.figure(figsize=(17,10))
plt.xlim([0.0,1000])
plt.ylim([0.0,30000])

plt.hist(x, num_bins)
plt.show()
# na histogrami możemy zwrócić uwagę iżjest wyjątkowo dużo aut które posiadają moc około 0 co jest prawdopodobnie spowodowane błędem.
# postaramy się usunąć te dane
mean = df['powerPS'].mean()
df['C3'] = df['powerPS'].apply(lambda x: 'small' if x < 0.01*mean else ('large' if x > 100 * mean else 'ok'))
df = df[df['C3']=='ok']
#usunołem dane z błędną 
#wartości zerowe został usunięte, jak i zarówno znaczne odchylenia w prawą stronę.
#mimo samochodów sportowych o najwyższej klasie i cenie raczej nie mają możliwości przekroczyć poziomu 100*średnia
df.groupby(by='vehicleType')['price'].count()
df[df['vehicleType'].isnull()].price.count()
#kategorii jest dużo, kolumna się nadaje do modelu.
#również znajdują sie braki w danych.
df['vehicleType']
df['vehicleType'] = df['vehicleType'].fillna('NAN')
lb_make = LabelEncoder()
df['vehicleType'] = lb_make.fit_transform(df['vehicleType'])
df.groupby(by='fuelType')['price'].count()
df[df['fuelType'].isnull()].price.count()
Ser = pd.Series(df.groupby(by='fuelType')['price'].count())
Ser = Ser.where(lambda x : x>4000).dropna()
a = list(Ser.index)
a
df[df['fuelType'].isin(a)]
#kategorii jest dużo, kolumna się nadaje do modelu.
#również znajdują sie braki w danych.
df['fuelType'] = df['fuelType'].fillna('NAN')
lb_make = LabelEncoder()
df['fuelType'] = lb_make.fit_transform(df['fuelType'])
df['brand'].unique()
europeanBrands = ['volkswagen','audi', 'skoda', 'bmw', 'peugeot','renault','mercedes_benz','seat','fiat','opel','mini','smart','alfa_romeo','volvo','lancia',
'porsche','citroen','dacia','trabant','saab','jaguar','rover','land_rover','lada']
americanBrands = ['chrysler','chevrolet','ford','jeep']
asianBrands = ['daewoo','daihatsu','toyota','mazda','kia','suzuki','mitsubishi','subaru','nissan','honda','hyundai']
otherBrands = ['sonstige_autos']
df['brand'] = df['brand'].apply(lambda x: 0 if x in europeanBrands else (1 if x in americanBrands else (2 if x in asianBrands else 3)))
df['brand'].head(20)
x = df['brand']
num_bins = 4

fig = plt.figure(figsize=(25,20))
plt.xlim([0,3])
plt.ylim([0,320000])

plt.hist(x, num_bins)
plt.yticks(np.arange(0, 320000, step=10000))
plt.grid()
plt.show()
df['brandOHE'] = df['brand'].apply(lambda x: [1,0,0,0] if x == 0 else ([0,1,0,0] if x == 1 else ([0,0,1,0] if x == 2 else [0,0,0,1])))
df['brandOHE'].head(20)
zliczeniaA = df['yearOfRegistration'].count()
df['monthsProd'] = df['monthOfRegistration'].apply(lambda x: 'small' if x < 0 else ('large' if x > 11 else 'ok'))
df = df[df['monthsProd'] == 'ok']
df['yearsProd'] = df['yearOfRegistration'].apply(lambda x: 'small' if x < 1886 else ('large' if x > 2019 else 'ok'))
df = df[df['yearsProd'] == 'ok']
zliczeniaB = df['yearOfRegistration'].count()
((zliczeniaA - zliczeniaB) / zliczeniaA) * 100
dates_dict = dict(year=df['yearOfRegistration'].values, month=df['monthOfRegistration'].values+1, day=df['yearOfRegistration']-df['yearOfRegistration']+1)
df['dateProd'] = pd.to_datetime(dates_dict)
df['dateProd'].head()
df['lastSeen'] = pd.to_datetime(df['lastSeen'])
df['vehicleAgeInYearsSinceLastSeen'] = df['lastSeen'] - df['dateProd']
df['vehicleAgeInYearsSinceLastSeen'] = df['vehicleAgeInYearsSinceLastSeen']/np.timedelta64(1, 'Y')
df['vehicleAgeInYearsSinceLastSeen'].head()
x = df['vehicleAgeInYearsSinceLastSeen']
num_bins = 120

fig = plt.figure(figsize=(25,20))
plt.xlim([-5,120])
plt.ylim([0.0,25000])

plt.hist(x, num_bins)
plt.xticks(np.arange(-5, 120, step=5))
plt.grid()
plt.show()
df['vehicleAgeUnderMin'] = df['vehicleAgeInYearsSinceLastSeen'].apply(lambda x: 'small' if x < 0 else 'ok')
df = df[df['vehicleAgeUnderMin'] == 'ok']
x = df['vehicleAgeInYearsSinceLastSeen']
num_bins = 120

fig = plt.figure(figsize=(25,20))
plt.xlim([-5,120])
plt.ylim([0.0,25000])

plt.hist(x, num_bins)
plt.xticks(np.arange(-5, 120, step=5))
plt.grid()
plt.show()
end_data = df[['price','kilometer','notRepairedDamage','gearbox','powerPS','vehicleType','fuelType','brand','vehicleAgeInYearsSinceLastSeen']]
end_data.to_csv('autos_reworked.csv', index=False)
end_data.head()
df_n = pd.read_csv('../working/autos_reworked.csv')
df_n.head(5)
df_n.corr()
corr = df_n.corr()
f, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5)
X_train, X_test, y_train, y_test = train_test_split(df_n[['kilometer','notRepairedDamage','gearbox','powerPS','vehicleType','fuelType','brand','vehicleAgeInYearsSinceLastSeen']],
df_n['price'],test_size = 0.25)
print(X_train, y_train)
df_n.dtypes
target = df_n.pop('price')
target.head(2)
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices((df_n.values, target.values))
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))
train_dataset = dataset.shuffle(len(df)).batch(1)
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model
model = get_compiled_model()
model.fit(train_dataset, epochs=15)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
model.fit(X_train,y_train)
pred = model.predict(X_test)
r2_score(y_test, pred)
model.fit(X_train,y_train)

