# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
data = pd.read_csv ('../input/2015.csv') #datasetimizi data adlı değişkene atadık
data.info() #datamız ile ilgili genel bilgileri gözden geçirdik
data.corr() #datamızın korelasyonuna baktık 
f,ax = plt.subplots(figsize=(18, 18)) #Şimdiki bilgilerimle bunu kendi başıma yazamam ama ilk satırı yazmadığımzı zaman kod hata veriyor,ax'in tanımlanmadığını söylüyor. İlk satırda muhtemelen ax'i tanımlıyoruz ve karelerin boyutunu belirtiyoruz.
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10) #İlk 10 elemanın bilgilerini aldık, eğer data.head() şeklinde yazsaydık ilk 5 elemanı verirdi
data.columns #Sütunlra bakmak için kullandık.
data.columns = [each.split()[0] +"_" + each.split()[1] if (len(each.split())>1) else each for each in data.columns]
data.columns
# Line plot 
# color = renk , label = etiket , linewidth = çizgi kalınlığı , alpha = opaklık , grid = ızgara, linestyle = çizgi stili

data.Freedom.plot (kind = 'line', color = 'm', label = 'Freedom', linewidth = 1, alpha = 0.7 , grid = True, linestyle = ':'  ) 
data.Happiness_Rank.plot (kind = 'line', color = 'r', label = 'Happiness_Rank', linewidth = 1, alpha = 0.7 , grid = True, linestyle = '-.'  ) #Bu iki satırda x ve y eksenlerine datamızının hangi kolonlarını koyacağımızı seçtik ve gerekli ayarlamaları yaptık
plt.legend(loc='upper left') #Bilgilendirme etiketini nereye koyacağımızı seçtik
plt.xlabel('x axis')
plt.ylabel('y axis') #x ve y eksenlerine isim verdik
plt.title('Freedom and Happiness Rank Relationship') #Grafiğimize isim verdik 
plt.show() #Grafiğimizi görmek istiyorsak bunu kesinlikle yazmamız gerekiyor


# Scatter plot

data.plot(kind ='scatter', x ='Family', y = 'Freedom', alpha = 0.5, color = 'red')
plt.xlabel('Family')
plt.ylabel('Region')
plt.title('Family-Region')
plt.show()
# Histogram 

data.Happiness_Score.plot(kind = 'hist', bins = 50 )
plt.show()

dictionary = {'Fruits':['Apple', 'Orange','Pomegranate'],'Vegatables': ['Onion', 'Potato', 'Cucumber']} #Dictionary adında bir sözlük oluşturduk ve Fruits ve Vegatabales adında iki anahtar belirledik. İlgili değişkenleri ise bu anahtarların içine yazdık.
print(dictionary.keys())
print(dictionary.values())

dictionary['Fruits'] = 'Apple'
print(dictionary)
dictionary['Dessert'] = 'IceCream'
print(dictionary)
del dictionary['Dessert']
print(dictionary)
print('Fruits'in dictionary)
dictionary.clear()
print(dictionary)

x = data['Family'] >0.8
data[x]
data[(data['Family']>0.8) & (data['Happiness_Score']>6)] #datamızın içindeki Family ve Happiness_Score adlı sütunların içindeki bilgilerden istediğimiz aralıkta olanları aldık.
print(data.Region)
liste = data.Region

we = 0
ssa = 0
for each in liste :
    if(each == 'Western Europe'):
        we = we + 1 
    elif(each == 'Sub-Saharan Africa'):
        ssa = ssa + 1
    else:
            continue
print(we)   
print(ssa)
            

def tuble_ex():
    """yorum satırı"""
    t = (1,2,3)
    return t 
print(tuble_ex())
x = data.Happiness_Score
def f():
    y = 2 * x
    return y
print(f())


def cember_cevre(r):
    def carp(pi=3.14):
        sonuc = 2 * pi
        return sonuc 
    return carp()*r
print(cember_cevre(2))
    
    
def f(*args):
    for i in args:
          print(i)
f(10)  

def k(**kwargs):
    for key, value in kwargs.items():
        print(key, " ", value)
k(country = 'spain', capital = 'madrid', populaiton = 123456)      

        
number_list = [1,2,3]
y = map(lambda x:x**2,number_list)
print(list(y))
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
num1 = [1,2,3]
num2 = [i+1 for i in num1]
print(num2)
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i - 5 if i < 7 else i + 5 for i in num1]
print(num2)
threshold = sum(data.Family) / len(data.Family)
data["Family_Size"] = ["big" if i> threshold else "low" for i in data.Family]
data.loc[:20,["Family_Size","Family"]]
print(threshold)
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data_world = pd.read_csv ('../input/world-happiness/2015.csv')
data_world.info()
data.head()
data.tail()
data.columns
data.shape
data.info()
print(data['Type 1'].value_counts(dropna = False)) #Type1 kolonunda tek tek kaç eleman olduğunu gösterir.
data.describe()
data.boxplot(column = 'Speed', by = 'Legendary')
data_new = data.head()
data_new
melted = pd.melt(frame = data_new, id_vars = 'Name', value_vars = ['Attack','Defense'])
melted
melted.pivot(index = 'Name', columns = 'variable', values = 'value')
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2],axis = 0, ignore_index = True)
conc_data_row
country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col =[country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df


df["capital"] = ["madrid","paris"]
df
df["income"] = 0
df
data1 = data.loc[ : , [ "Attack" , "Defense" , "Speed"]]
data1.plot()
plt.show()
data1.plot(subplots = True )
plt.show()
data1.plot(kind = "scatter", x="Attack", y ="Defense")
plt.show()
data1.plot(kind = "hist", y = "Attack", bins = 50 , range = (0,250), normed = True)

fig,axes = plt.subplots(nrows = 2, ncols = 1)
data1.plot (kind = "hist", y = "Defense", bins = 50, range = (0,250), normed = True, ax = axes[0])
data1.plot (kind = "hist", y = "Defense", bins = 50, range = (0,250), normed = True, ax = axes[1], cumulative = True )
plt.savefig ('graph.png')
plt

time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1]))

datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
import warnings 
warnings.filterwarnings("ignore")

data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object

data2 = data2.set_index("date")
data2
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")