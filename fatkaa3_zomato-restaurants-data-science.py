# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# csv dosyamızı okuyoruz.
data = pd.read_csv("../input/zomato.csv", encoding ='iso-8859-9')

# kaç satır, kaç sütun 
data.shape
# Sütun isimleri
data.columns
# datanın ilk 5 satırı
data.head()
# data nın son 5 satırı
data.tail()
# data hakkında genel bilgiler 
data.info()
# data da boş alan var mı?
data.isnull().sum()
print("Dataset içerisindeki bulunan ülke kodları:\n".upper())
print(len(data['Country Code'].unique()), "farklı değer var")
print(data['Country Code'].unique())
print("\nDataset içerisindeki bulunan şehirler:\n".upper())
print(len(data['City'].unique()), "farklı değer var")
print(data['City'].unique())
print("\nDataset içerisindeki bulunan Mutfak Türleri:\n".upper())
print(len(data['Cuisines'].unique()), "farklı değer var")
print(data['Cuisines'].unique())
print("\nDataset içerisindeki bulunan Restoranda ödenen para birimleri:\n".upper())
print(len(data['Currency'].unique()), "farklı değer var")
print(data['Currency'].unique())
print("\nDataset içerisindeki bulunan Derecelendirme renkleri:\n".upper())
print(len(data['Rating color'].unique()), "farklı değer var")
print(data['Rating color'].unique())
print("\nDataset içerisindeki bulunan Derecelendirme metni:\n".upper())
print(len(data['Rating text'].unique()), "farklı değer var")
print(data['Rating text'].unique())
data['City'].value_counts(dropna = False)
data.describe()
data.boxplot(column = 'Aggregate rating')
data_new = data.head()
melted = pd.melt(frame = data_new, id_vars = 'Restaurant Name', value_vars = ['Aggregate rating','Rating color','Rating text'])
melted
melted.pivot(index = 'Restaurant Name', columns = 'variable', values = 'value')
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1, data2], axis = 0, ignore_index = True)
conc_data_row
data1 = data['Restaurant Name'].head()
data2 = data['Votes'].head()
conc_data_col = pd.concat([data1, data2], axis =1)
conc_data_col
assert data['Cuisines'].notnull().all() 
# data['Cuisines'].dropna(inplace = True) # missing dataları silmek için 
data['Cuisines'].fillna('Unknown', inplace = True)
assert data['Cuisines'].notnull().all()
print("Mutfak tipi bilinmeyen veriler".upper())
print("Sayısı: ", len(data[data['Cuisines'] == 'Unknown']))
data[data['Cuisines'] == 'Unknown']
name = list(data['Restaurant Name'].head()) # Restoran isimlerının ilk 5 öğesin alıp listeye dönüştürdük
address = list(data['Address'].head()) # Restoran adreslerinin ilk 5 öğesin alıp listeye dönüştürdük
list_label = ['name','address'] # sütun isimlerimiz
list_col = [name, address] # içerisine alacağımız değerler 
zipped = list(zip(list_label, list_col)) # sütun isimleri ve değerleri birleştirdik
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
df['number_of_employees']= 5
df
average_cost_for_two = list(data['Average Cost for two'].head())
votes = list(data['Votes'].head())
df['average_cost_for_two'] = average_cost_for_two
df['votes'] = votes
df
data.plot()
plt.show()
data.plot(subplots = True)
plt.show()
data.plot(kind = "hist", y = 'Votes',bins = 20, range = (200,600), normed = True)
plt.show()
# histogram ile cumulative olmayan ve cumulative olan 
fix, axes = plt.subplots(nrows = 2, ncols = 1)
data.plot(kind = 'hist', y = 'Votes',bins = 50, range = (200,600), normed = True , ax = axes[0])
data.plot(kind = 'hist', y = 'Votes',bins = 50, range = (200,600), normed = True , ax = axes[1], cumulative = True)
plt.savefig('graph.png')
plt.show()
year_of_establishment_list = ["1998-03-10", "1999-03-05", "1999-04-21", "1998-11-30", "1999-04-01"]
datetime_object = pd.to_datetime(year_of_establishment_list) # datetime yaptık.
print(type(datetime_object))
df['year_of_establishment_list'] = datetime_object
df
df = df.set_index("year_of_establishment_list")
df
df.resample("A").mean()
df.resample("M").mean()
df.resample("M").first().interpolate("linear")
df.resample("M").mean().interpolate("linear")
df['index']=[1,2,3,4,5]
df = df.set_index("index")
df
df.loc[1:3,"name":"number_of_employees"]
df.loc[3:1:-1,"name":"number_of_employees"]
df.loc[1:3,"number_of_employees":]
boolean = df.votes > 300
df[boolean]
filtre1 = df.votes < 400
filtre2 = df.average_cost_for_two > 1200
df[filtre1 & filtre2]
df.average_cost_for_two[df.votes >350]
def div(f):
    return f+2
df.number_of_employees.apply(div)
df.number_of_employees.apply(lambda x : x+2)
df['new'] = df.average_cost_for_two / df.number_of_employees
df