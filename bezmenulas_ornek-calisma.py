# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls","../input"]).decode("utf-8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon.csv')
data.info()
data.corr() # değerler arası ilişkileri belirmek 

#Korelasyon, olasılık kuramı ve istatistikte iki rassal değişken arasındaki doğrusal ilişkinin yönünü ve gücünü belirtir.
# correlation map

f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.2f',ax=ax)

# figsize=(18,18) --> Boyutu ayarladık

# annot = True --> kutu içinde değerlerin gösteriyor.

# linewidths=.5 --> kutular arası boşluk belirtiyor.

# fmt='.1f' --> Ondalıklı sayılarda virgülden sonra kaç basamak yazılacağını belirtiyor.



plt.show()
data.columns
data.Attack.plot(kind='line') # Line türü grafik çizdirdik.

plt.show()
data.Speed.plot(kind = 'line', color='g', label='Speed', linewidth=1, alpha=0.5, grid=True, linestyle=':' )

data.Defense.plot(color='r', label='Defence', linewidth=1, alpha=0.5, grid=True, linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
# x = attack , y = defense

data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color='red')

# plt.scatter(data.Attack,data.Defense)

plt.xlabel('Attack')

plt.ylabel('Defense')

plt.title('Attack Defense Scatter Plot')

plt.show()



#plt.scatter(data.Attack,data.Defense, alpha=0.5, color='red')
data.Speed.plot(kind='hist', bins=50, figsize=(12,12))

plt.show()



#plt.hist(data.Speed, bins=50)
data.Speed.plot(kind='hist', bins=50, figsize=(12,12))

plt.clf()

# plt.clf() --> Clear
dictionary = {'Fire':'Charmander', 'Grass':'Bulbasaur', 'Water':'Squirtle'}

print(dictionary.keys())

print(dictionary.values())
dictionary['Fire'] = "Charizard" # deger değistirme

print(dictionary)



dictionary['Bug'] = "Butterfree" # yeni ekleme

print(dictionary)



del dictionary['Grass'] # silme

print(dictionary)



print('Bug' in dictionary)



dictionary.clear() # tamamen siliyor.

print(dictionary)



del dictionary # dictionary siler.

# print(dictionary) hata verir çünkü silmiştik.
series = data['Defense']

print(type(series))

data_frame = data[['Defense']]

print(type(data_frame))
# Filtering pandas data frame

x = data['Defense'] > 200 # defansı 200'den büyük olanlar

data[x]
data[np.logical_and(data['Defense']>200, data['Attack']>100)]

# savunması 200'den büyük olanlar ve saldırısı 100'den büyük olanlar.

# data[(data['Defense']>200) & (data['Attack']>100)]
data["Type 1"].unique()

# Type 1'de bulunan bütün türleri yazdırdık.
data[(data['Type 1'] == 'Dragon')] 

# Type 1, Dragon olanları aldık.
threshold = sum(data.Speed)/len(data.Speed) # hızların ortalaması

print("threshold = ",threshold)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]]
#List Comprehension

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i<7 else i+5 for i in num1]



num3 =[i*9 if i == 10 else i+10000 if i==15 else i*0 if i==5 else i-90 for i in num1]

print("Num1 : ",num1)

print("Num2 : ",num2)

print("Num3 : ",num3)

# [islem1 if koşul1 else islme2 if koşul2 else işlem3 if koşul3 else islem for döngüsü]
data.head() # head shows first 5 rows

# data.head(7)  show first 7 rows
data.tail() # tail shows last 5 rows
data.shape # (800,13) ---> (rows,colums)
data['Type 1'].value_counts()  # Type 1 türlerinden kaçar tane pokemon var
data.describe()
data.boxplot(column='Attack', by='Legendary')

plt.show()
data_new = data.head()

data_new
# melt() ----> Datayı şekillendiriyoruz. 

#Name sutunu kalsın. ----> id_vars

#Attack ve Defense, variable adında yeni bir sutuna ata ----> value_vars  

#Attack ve Defense değerlerini value adında yeni bir sutuna ata (value()) ----> value_vars   

melted = pd.melt(frame=data_new, id_vars='Name', value_vars=['Attack','Defense'])

melted
# Pivoting ----> melt ile yaptığımızı eski haline getiriyor.

melted.pivot(index='Name', columns='variable', values='value')
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2],axis = 0,ignore_index=True)

conc_data_row

# Satır olarak birleştirme
data1 = data['Attack'].head()

data2 = data['Defense'].head()

conc_data_col = pd.concat([data1,data2], axis=1)

conc_data_col

# Stusun olarak ekleme
data.dtypes
data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')

data.dtypes

# Type 1 = object -----> Type 1 = category çevirdik.

# Speed = int64 -----> Speed = float çevirdik.
data.info() # Type2 ----> 414 non-null object
data['Type 2'].value_counts(dropna = False)

# dropna = False ----> Kaç Nan değeri var onu veriyor. 386 Nan -> Mising value
data1 = data

data1["Type 2"].dropna(inplace = True)

# Type 2'si olmayan pokemonları attık.
assert 1==1 #kontrol ediyo. Bir şey döndürmezse doğrudur.
assert data['Type 2'].notnull().all() 

# Type 2'si olmayanlar atmışız.
data["Type 2"].fillna('empty',inplace = True) 

# Type2 empty ile doldurduk.
data.head()
assert data.columns[1] == 'Name' # bir şey döndürmicek

# assert data.columns[1] == 'HP' # hata verir.
data.Speed.dtypes
assert data.Speed.dtypes == np.float64
pokemons = ['Bulbasaur','Charmander']

attack = ['49','52']

list_label = ['pokemons','attack']

list_col = [pokemons,attack]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

# listtelerden -> dic -> dataframe yarattık.

df
df["Defense"] = ['49','43'] # yeni bir colums oluşturduk

df
# Broadcasting

df["Generation"] = 1 # Generation, columns bütün değerler = 1

df
data1 = data.loc[:,['Attack','Defense','Speed']]

data1.plot()

plt.show()
data.plot(subplots = True)

plt.show()
data1.plot(subplots=True)

plt.show()
data1.plot(kind='scatter', x = 'Attack', y = 'Defense')

plt.show()
data1.plot(kind='hist',y = 'Defense',bins=50,range=(0,250))

plt.show()



# range ----> 0'dan 250 kadar istedik.

# normed=True -----> normalize etme.
fig, axes= plt.subplots(nrows=2, ncols=1)

data1.plot(kind='hist',y='Defense',bins=50,range=(0,250),ax=axes[0])

data1.plot(kind='hist', y='Defense',bins=50,range=(0,250),ax=axes[1],cumulative= True)

plt.savefig('graph.png')

data.head()
time_list = ["1975-07-08","1980-04-17"]

print(type(time_list[1]))



datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
import warnings

warnings.filterwarnings('ignore')



data2 = data.head()

date_list = ["1975-07-08","1980-04-17","1980-05-17","1980-08-22","1990-04-29"]

datetime_objects = pd.to_datetime(date_list)

data2["date"] = datetime_objects



data2 = data2.set_index("date")

data2
print(data2.loc["1975-07-08"])
print(data2.loc["1980-05-17":"1990-04-29"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")

# lineer olarak dolduruyor.
data = data.set_index('#')

data.head()

# index 0'dan başlıyodu şimdi 1'den başlıyor.
data['HP'][1]
data.HP[1]
data.loc[1,['HP']]
data[['HP','Attack']]
print(type(data['HP'])) # ----> series

print(type(data[['HP']])) # -----> DataFrame
data.loc[1:10,"HP":"Defense"] # ----> 1'den 10'a kadar ve HP'den Defense kadar verdi.
data.loc[10:1:-1,"HP":"Defense"] #-----> ters yazdırdık.
data.loc[1:10,"Speed":] # -----> 1'den 10'a kadar ve Speed'den sonrakileri aldık
boolean = data.HP > 200

data[boolean]
first_filter = data.HP > 150

second_filter = data.Speed > 35

data[first_filter & second_filter]
data.HP[data.Speed<15]
def div(n):

    return n/2

data.HP.apply(div)
data.HP.apply(lambda n: n/2)
data["total_power"] = data.Attack + data.Defense

# total_power adında yeni bir sutun oluşturduk.

data.head()
print(data.index.name) # ----->  data.index.name = #



data.index.name = "index_name" # ismi değiştirdik. # yerine index_name yaptık

data.head()
data3 = data.copy() #  kopyaladık

data3.index = range(100,900,1) # 100 başlattık 900'e kadar birer birer arttırdık.  

data3.head()
data1 = data.set_index(['Type 1','Type 2'])

data1.head(100)



# Type 1 ve Type 2'yi index haline getirdik.
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
df.pivot(index="treatment",columns="gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)

df2
df
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
df.groupby("treatment").mean() 

# ortalamalarına göre grupladık.
df.groupby("treatment").age.max()

# maksimumlarına göre grupladık.
df.groupby("treatment")[["age","response"]].min()
df.info()