# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
##reading the data 
df = pd.read_csv('../input/BlackFriday.csv')
df.info()
df.head(10)
##Localizo las variables con valores nulos
df.isnull().sum()
##Les asigno valor 0, ya que de momento no sé si voy a necesitar esas columnas
df.fillna(value=0,inplace=True)
##varios items para un mismo User_ID, compruebo cuantos usuarios únicos
print("Length of the dataset: %d\nNumber of different User_ID: %d" % (len(df), len(df.User_ID.unique())))
sns.countplot (df['Gender'])
sns.boxplot(x="Gender", y="Purchase", data=df)
##Para poder trabajar mejor con esta variable la transformo a F=0 y M=1
gender = np.unique(df['Gender'])
gender
def map_gender(gender):
    if gender == 'M':
        return 1
    else:
        return 0
df['Gender'] = df['Gender'].apply(map_gender)
df.head()
sns.countplot (df['City_Category'])
sns.boxplot(x="City_Category", y="Purchase", data=df)
city_category = np.unique(df['City_Category'])
city_category
def map_city_category(city_category):
    if city_category == 'A':
        return 1
    elif city_category == 'B':
        return 2
    elif city_category == 'C':
        return 3
    else:
        return 0
df ['City_Category'] = df ['City_Category'].apply(map_city_category)
sns.countplot (df['Age'])
sns.boxplot(x="Age", y="Purchase", data=df)
##Para la variable Age me encuentro con rangos y el 55+, voy a limpiar esta variable.
age = np.unique(df['Age'])
age
def map_age(age):
    if age == '0-17':
        return 0
    elif age == '18-25':
        return 1
    elif age == '26-35':
        return 2
    elif age == '36-45':
        return 3
    elif age == '46-50':
        return 4
    elif age == '51-55':
        return 5
    else:
        return 6
df ['Age'] = df ['Age'].apply(map_age)
sns.countplot (df['Stay_In_Current_City_Years'])
sns.boxplot(x="Stay_In_Current_City_Years", y="Purchase", data=df)
##La variable Stay In Current City Years contiene el valor +4, lo transformo para que me devuelva solo 4
def map_stay(stay):
        if stay == '4+':
            return 4
        else:
            return int(stay)
#             current_years = stay
#             current_years = current_years.astype(int)
#             return current_years
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].apply(map_stay)
df.head()
df.describe()
##Escalado de valores númericos. Para trabajar los algoritmos escalo los valores númericos al intervalo entre 0 y 1 de las siguientes variables: Occupation; Product_Category_1; Product_Category_2; Product_Category_3 y Purchase
##al dataset resultante le voy a denominar scaleddf para no sobreescribir el original y poder seguir trabajando con ambos
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
dfScaled = mms.fit_transform(df[['Age', 'Gender', 'Stay_In_Current_City_Years','Occupation','Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']])
df[['Age', 'Gender', 'Stay_In_Current_City_Years','Occupation','Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']] = dfScaled
df[['Age', 'Gender', 'Stay_In_Current_City_Years','Occupation','Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']].describe()
##Errores en el campo de edad (no comprende el rango) y en el de Stay In Current City Years (no coge el 4+)
f, ax = plt.subplots(figsize=(12, 12))
corr = df.corr()
sns.heatmap(corr,annot=True,linewidths=.5, fmt= '.2f',mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(100,200, as_cmap=True), square=True, ax=ax)
plt.show()
#Para buscar el perfil medio combino las variables Gender y Marital_Status
df['combined_G_M'] = df.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)
print(df['combined_G_M'].unique())
sns.countplot(df['Age'],hue=df['combined_G_M'])
df2=df[['Purchase', 'User_ID']].groupby('User_ID').sum()
df2.head()
df2 = df2.rename(columns={'Purchase': 'totalPurchase'})
df2.head()
dfMerge = df.merge(df2, on='User_ID')
dfMerge.head()
dfMerge.describe()
sns.boxplot("totalPurchase", data=dfMerge)
sns.boxplot(x="Stay_In_Current_City_Years", y="totalPurchase", data=dfMerge)
sns.boxplot(x="Age", y="Purchase", data=df)
df3=df[['Purchase', 'User_ID']].groupby('User_ID').count()
df3.head()
df3 = df3.rename(columns={'Purchase': 'countPurchase'})
df3.head()
dfMerge2 = dfMerge.merge(df3, on='User_ID')
dfMerge2.head()
sns.boxplot("countPurchase", data=dfMerge2)
dfMerge2.describe()
sns.boxplot(x="Gender", y="totalPurchase", data=dfMerge2)
sns.boxplot(x="Gender", y="countPurchase", data=dfMerge2)
f, ax = plt.subplots(figsize=(13, 13))
corr = dfMerge2.corr()
sns.heatmap(corr,annot=True,linewidths=.5, fmt= '.2f',mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(100,200, as_cmap=True), square=True, ax=ax)
plt.show()
sns.boxplot(x="City_Category", y="totalPurchase", data=dfMerge2)
sns.boxplot(x="City_Category", y="countPurchase", data=dfMerge2)
df3=dfMerge2[['totalPurchase', 'Product_ID']].groupby('Product_ID').sum()
df3.head()
productos_mas_vendidos= df3.sort_values(by='totalPurchase', ascending=False)
productos_mas_vendidos= productos_mas_vendidos.head(15)
print(productos_mas_vendidos)
product_count = Counter (df.Product_ID)         
most_common_product = product_count.most_common(15)  
x,y = zip(*most_common_product)
x,y = list(x),list(y)
# visualization
plt.figure(figsize=(15,10))
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Product_ID')
plt.ylabel('Frequency')
plt.title('Most common 15 Product_ID')