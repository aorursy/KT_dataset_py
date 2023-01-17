import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/blore_apartment_data.csv")
df.head()
df.isnull().values.any()
df.isnull().sum()
df = df.dropna(how='any',axis=0)
df['Price'].value_counts()
PriceList = list(df['Price'])
def sep(li):

    newli1=[]

    newli2=[]

    for i in range(len(li)):

        text = li[i]

        head, sep, tail = text.partition('-')

        newli1.append(head)

        newli2.append(tail)

    return newli1,newli2
Min,Max= sep(PriceList)
def Converter(li):

    newli=[]

    for i in range(len(li)):

        if 'L' in li[i]:

            text = li[i]

            li[i] = li[i].replace('L',' ')

            li[i] = float(li[i])

            li[i] = li[i]*100000

            li[i] = int(li[i])

            newli.append(li[i])

            li[i] = str(li[i])

        elif 'K' in li[i]:

            text = li[i]

            li[i] = li[i].replace('K',' ')

            li[i] = float(li[i])

            li[i] = li[i]*1000

            li[i] = int(li[i])

            newli.append(li[i])

            li[i] = str(li[i])

        elif 'Cr' in li[i]:

            text = li[i]

            li[i] = li[i].replace('Cr',' ')

            li[i] = float(li[i])

            li[i] = li[i]*10000000

            li[i] = int(li[i])

            newli.append(li[i])

            li[i] = str(li[i])

        else:

            newli.append(li[i])

    return newli
MinRange = Converter(Min)

MaxRange = Converter(Max)     
df['MinRange'] = MinRange

df['MaxRange'] = MaxRange
df[["MinRange", "MaxRange"]] = df[["MinRange", "MaxRange"]].apply(pd.to_numeric)
col = df.loc[: , "MinRange":"MaxRange"]

df['AveragePrice'] = col.mean(axis=1)
df = df.drop(['MaxRange','MinRange','Price'], axis=1)
df.head()
def AreaConverter(li):

    newli=[]

    for i in range(len(li)):

        if 'sq.ft' in li[i]:

            text = li[i]

            li[i] = li[i].replace('sq.ft','')

            newli.append(li[i])

        else:

            newli.append(li[i])

    return newli
AreaList = list(df['Area'])

AreaWithoutUnit = AreaConverter(AreaList)
AreaWithoutUnit 
Min,Max= sep(AreaWithoutUnit)
df['MinArea'] = Min

df['MaxArea'] = Max
df['AverageArea'] = df[['MinArea','MaxArea']].mean(axis=1)
df = df.drop(['MinArea','MaxArea','Area'], axis=1)
df.head()
UnitTypeList= list(df['Unit Type'])
def BHK(li):

    newli = []

    for i in range(len(li)):

        if 'Plot' in li[i]:

            li[i] = str("0 Not:BHK/RK ") + li[i]

            newli.append(li)

        else:

            newli.append(li)

    return newli
BHK1 = BHK(UnitTypeList)
def Unitsep(li):

    newli1=[]

    newli2=[]

    for i in range(len(li)):

        text = li[i]

        head, sep, tail = text.partition(' ')

        newli1.append(head+sep)

        newli2.append(tail)

    return newli1,newli2
NoUnitType,Uni2 = Unitsep(UnitTypeList)

BHKorRK,UnitType1 = Unitsep(Uni2)
df['UnitNo'] = NoUnitType

df['BHKorRK'] = BHKorRK

df['UnitType']= UnitType1
df = df[df.UnitNo != 'Studio ']

df= df[df.UnitNo != 'Apartment']

df= df[df.UnitType != 'BHK Apartment']
df = df.drop(['Unit Type'], axis=1)
df.head()
df['UnitNo'].value_counts()
df = df.replace('4+ ','4.5')

df.UnitNo = df.UnitNo.astype(float)

df['UnitNo'].value_counts()
df.head()
dataset = df[['names','UnitNo','BHKorRK','UnitType','AverageArea','AveragePrice']]
dataset.head()
X = dataset.iloc[:, 4:5].values

y = dataset.iloc[:, 5:].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Price vs Area (Training set)')

plt.xlabel('Area')

plt.ylabel('Price')

plt.show()
plt.scatter(X_test, y_test, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Price vs Area (Test set)')

plt.xlabel('Area')

plt.ylabel('Price')

plt.show()
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)