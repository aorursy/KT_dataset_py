from matplotlib import style

import matplotlib.pyplot as plt

style.use('ggplot')

%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns



from sklearn import model_selection

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
df=pd.read_csv('../input/autos.csv', encoding='ISO-8859-1')

df.head()
del_cols = ['dateCrawled','dateCreated','nrOfPictures','lastSeen','postalCode','abtest']

df.drop(del_cols,axis=1,inplace=True)
df['seller'] = df['seller'].replace('privat','private')

df['seller'] = df['seller'].replace('gewerblich','commercial')



df['offerType'] = df['offerType'].replace('Angebot','offer')

df['offerType'] = df['offerType'].replace('Gesuch','petition')



df['gearbox'] = df['gearbox'].replace('manuell','manual')

df['gearbox'] = df['gearbox'].replace('automatik','automatic')



df['fuelType'] = df['fuelType'].replace('benzin','petrol')

df['fuelType'] = df['fuelType'].replace('andere','Others')

df['fuelType'] = df['fuelType'].replace('elektro','electric')



df['notRepairedDamage'] = df['notRepairedDamage'].replace('nein','no')

df['notRepairedDamage'] = df['notRepairedDamage'].replace('ja','yes')



df['vehicleType'] = df['vehicleType'].replace('kleinwagen','small_car')

df['vehicleType'] = df['vehicleType'].replace('andere','others')

df['vehicleType'] = df['vehicleType'].replace('kombi','wagon')
df= df[(df.yearOfRegistration <= 2016)& (df.yearOfRegistration >= 1975) & (df.price>=100) & (df.price<=150000) & (df.powerPS > 50)] 

df.shape
df = df.dropna()

df = df.reset_index()
df.rename(columns = {'name':'NAME','seller':'SELLER','price':'PRICE','model':'MODEL','brand':'BRAND','gearbox':'GB','kilometer':'DISTANCE','fuelType':'FT','offerType':'OT','vehicleType':'VT','powerPS':'PPS','notRepairedDamage':'NRD'}, inplace = True)
df['years_old'] = df['yearOfRegistration'].apply(lambda x: 2016-x)
plt.figure(figsize=(12,10))

plt.subplot(2,2,1)

sns.countplot(df['SELLER'])

plt.subplot(2,2,2)

sns.countplot(df['OT'])

plt.subplot(2,2,3)

sns.countplot(df['VT'])

plt.subplot(2,2,4)

sns.countplot(df['GB'])

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111)





ax.set_xlabel('BRAND')  #X-axis label

ax.set_ylabel('PRICE OF CAR') #Y-axis label

ax.set_title("Brand wise Sum of price") #Chart title



F = df.groupby('BRAND').PRICE.sum()

F.plot(kind='bar',figsize=(20,10),rot=-90,color='yellow')
df['BRAND'].value_counts().plot(kind='bar',figsize=(20,10),rot=-30,color='blue')
price_year = df[['PRICE']].groupby(df['years_old']).mean()

price_year.plot(kind='bar',figsize=(20,10),rot=-30)
distance_year = df[['DISTANCE']].groupby(df['years_old']).mean()

distance_year.plot(kind='bar',figsize=(20,10),rot=-30,color='green')
df = pd.get_dummies(df, columns=["GB"], prefix=["GB"])

df = pd.get_dummies(df, columns=["FT"], prefix=["FT"])

df = pd.get_dummies(df, columns=["MODEL"], prefix=["MODEL"])

df = pd.get_dummies(df, columns=["VT"], prefix=["VT"])

df = pd.get_dummies(df, columns=["BRAND"], prefix=["BRAND"])

df = pd.get_dummies(df, columns=["NRD"], prefix=["NRD"])

df = pd.get_dummies(df, columns=["DISTANCE"], prefix=["DISTANCE"])



#df = pd.get_dummies(df, columns=["SELLER"], prefix=["SELLER"])

#df = pd.get_dummies(df, columns=["OT"], prefix=["OT"])
from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()

cat= df.columns[df.dtypes==object]

for col in cat:

    df[col]= le.fit_transform(df[col])
df["PRICE_NEW"] = np.log1p(df["PRICE"])

Y = df['PRICE_NEW']

name = df['NAME']
drop_cols = ['NAME','PRICE','PRICE_NEW','SELLER','OT','monthOfRegistration']

df = df.drop(drop_cols,axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.3)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
print('LR model is starting...')

from sklearn.metrics import mean_squared_error, r2_score



LRmodel = LinearRegression()

LRmodel.fit(X_train, y_train)

y_pred = LRmodel.predict(X_test)



print('========================')

print('MSE:',mean_squared_error(y_test,y_pred))

print('R2 score:',r2_score(y_test, y_pred))

print('========================')



print('LR model is finishing...')