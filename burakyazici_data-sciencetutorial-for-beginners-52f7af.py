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
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon.csv')# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python   
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)  
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data.info()
data.corr()    #  bir evi oda sayısı artarsa fiyatı dar artar        (yani 1,doğru orantı )
               #  bir ev şehir merkezinden uzaklaşırsa fiyat azalır  (yani -1,ters orantı)
               # aşağıdaki tabloda feature lerin bir bir leri ile olan ilişkileri var.
               # ilişki ne kadar 1 yakınsa o oranda birbirlerini etkiliyolarsa (x 0.5 artıyorsa y de 0.5 artar )
               # ilişki ne kadar -1 yakınsa o oranda birbirlerini etkiliyolarsa (x 0.5 artıyorsa y de 0.5 azalır )
               # eğer sıfır çıkar sa 2 özellik arasında hiçbir bağlantı yok demektir ( elma ile uçağın karşılaştırmak gibi) 


#correlation map   #feature ler arasındaki ilişkiyi sağlayan parametrelerdendir.. yukarda metot hakkında bilgi verdik
f,ax = plt.subplots(figsize=(18, 18)) #(18,18 ) aşağıdaki karelerin ölçüleri
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)  # bu kısımda başlığın özellikleri
plt.show()
data.head(10)
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')                                                        # burda id ler 2 feature karşılaştırıyoruz correlation bulmak için
plt.title('Line Plot')            # title = title of plot                     
plt.show()                      # x eksenindeki değerler pokemonların id leri
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red') # plt.scatter(data.Attack,data.defanse,colar="red",alpha=0.5) bu 2 işlem de aynı işi görür
plt.xlabel('Attack')              # label = name of label                    
plt.ylabel('Defence')  #                        burda 2 feature birbbirine göre kıyaslıyoruz c
plt.title('Attack Defense Scatter Plot')            # title = title of plot
# Histogram
# bins = number of bar in figure
data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))  # bu verideki hızların frekansı buluruz x ekseni hız y ekseni o hızdan kaç pokemon olduğunu verir
plt.show()                                                  # bin ise grafikte çubuklardan kaç tane olucağını söyler
# clf() = cleans it up again you can start a fresh
data.Speed.plot(kind = 'hist',bins = 50)
plt.clf()  #bu metot çizdir miş olduğumuz histogramı siler 
# We cannot see plot due to clf()
#create dictionary and look its keys and values
dictionary = {'spain' : 'madrid','usa' : 'vegas'}
print(dictionary.keys())
print(dictionary.values())            #DSFAAAAAAAFSDA
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['spain'] = "barcelona"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['spain']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)

# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv('../input/pokemon.csv')

series = data['Defense']        # data['Defense'] = series  # pandas da veri oluşturur 2 tip veri oluşturabiliriz
print(type(series))
data_frame = data[['Defense']]  # data[['Defense']] = data frame
print(type(data_frame))

# Comparison operator
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)
data.columns
# 1 - Filtering Pandas data frame
x = data['Defense']>200     # There are only 3 pokemons who have higher defense value than 200
data[x]                     # burda pokemon versinin içindeki "defanse" feture nda 200 den büyük olanları x değişkeninw atıyoruz
y = data.Defense>200
data[y]
# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100
data[np.logical_and(data['Defense']>200, data['Attack']>100 )]
# np.logical_and metodu numpy kutuphanesinde vardır
#ve operoterünün işini görür defans >200 ve attack> 100 den büyük olan verileri gösterir. 
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['Defense']>200) & (data['Attack']>100)]
# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1 
print(i,' is equal to 5')
# Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ',i)
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):                #  enumerate metodu ile listenin hem index sine hem değerlerine ulaşırız
    print(index," : ",value)
print('')   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():               #  dictionary lerde  key ve value ulaşmak için dictionary.items() metodu kullanır
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['Attack']][0:1].iterrows(): 
    print(index," : ",value)                      #  data nın içindeki attakc feature 0-1 arasondaki idex ve value değerlerini yazar
                                                  #  aralığı [0:1] yaptığımız için ilk indexi alır


# example of what we learn above
def tuble_ex():
    """ return defined t tuble"""
    t = (1,2,3)        
    return t           # tuble oluşturduk ve dönen değerin herbirisini farklı değişkene atadık bu tuble lın özelliği  
a,b,c = tuble_ex()     # avantajı= dönen değerde istemediğimiz bir değişken olurssa kullanmak zorunda değiliz   
print(a,b,c)
# guess print what
x = 2
def f():
    x = 3
    return x
print(x)      # x = 2 global scope
print(f())    # x = 3 local scope
# What if there is no local scope
x = 5
def f():
    y = 2*x        # there is no local scope x
    return y
print(f())         # it uses global scope x
# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
# How can we learn what is built in scope
import builtins
dir(builtins)
#nested function
def square():
    """ return square of value """
    def add():
        """ add two local variable """
        x = 2
        y = 3
        z = x + y
        return z
    return add()**2
print(square())    
# default arguments
def f(a, b = 1, c = 2):
    y = a + b + c
    return y
print(f(5))
# what if we want to change default arguments
print(f(5,4,3))
# flexible arguments *args
def f(*args):           # args her defasında farklı metot yazmaktansa args kullanımı daha mantıklı
    for i in args:      # çünkü metodu çağırdığımız istediğimiz kadar parametre girebilir herhangibi bir kısıtlma yok.
        print(i)
f(1)
print("")
f(1,2,3,4)
# flexible arguments **kwargs that is dictionary
def f(**kwargs):                                    # bu args dictionary için kullanılan özel bir gösterimi
    """ print key and value of dictionary"""
    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop
        print(key, " ", value)
f(country = 'spain', capital = 'madrid', population = 123456)   # burda metodu çağırıyoruz
# lambda function
square = lambda x: x**2     # where x is name of argument  # lamda ile metotları tek satırda yazabiliriz
print(square(4))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(1,2,3))
number_list = [1,2,3]
y = map(lambda x:x**2,number_list) #map metodu listenin içindeki değerleri tek tek lambda fonk da çalıştırıyor
print(list(y))
# iteration example
name = "ronaldo"
it = iter(name)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration

# zip example
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)             # (1,5),(2,6)..... diye birleştirii
z_list = list(z)     #birleştirdiğimiz veriyi liste türüne çevirip yazdırıyoruz
print(z_list)
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)         #yukardaki listeyi ayırmak için kulllanılır
print(un_list2)
print(type(un_list2))
# Example of list comprehension
num1 = [1,2,3]
num2 = [i + 1 for i in num1 ] # num1 listesinin içindeki her bir değeri artırarak num2 listesini oluşturuyoruz
print(num2)
# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
# lets return pokemon csv and make one more list comprehension example
# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

threshold = sum(data.Speed)/len(data.Speed)   # data speed feture deki verilerin toplamı / uzunluğu ile hızların ortalamasını buluyoruz  

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]  #döngü speed featurede dönsün ortalama dan küçük ise yavaş büyük ise hızlı atasın ve yeni bir özellik olutursun

data.loc[:10,["speed_level","Speed"]] # we will learn loc more detailed later  #ilk 10 değere bakıyoruz 2 colums için
#data[['Speed',"speed_level"]][0:10]   # üstteki kod la aynı işi görür

# data yı çağırmak =data.speed veya data["speed"] şeklinde olur 2.si genellikle name arasında boşluk var olursa kullanılır
data = pd.read_csv('../input/pokemon.csv')
#data.head()  # head shows first 5 rows

# tail shows last 5 rows
data.tail()
# columns gives column names of features
data.columns
# shape gives number of rows and columns in a tuble
data.shape # kaç satır ve sutun olduğunu gösterir
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()
# For example lets look frequency of pokemom types
print(data['Type 1'].value_counts(dropna =False))  # if there are nan values that also be counted
# As it can be seen below there are 112 water pokemon or 70 grass pokemon

# value_counts() metodu o feature deki verilerin sıklığını verir
# For example max HP is 255 or min defense is 5
data.describe() #ignore null entries
# For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Attack',by = 'Legendary')
# Firstly I create new data from pokemons data to explain melt nore easily.
data_new = data.head()    # I only take 5 rows into new data
data_new
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted

#melt metodu yukarda olan veriyi parçalamaya yarar burda id olarak name seçiyoruz ondan sonra istediğimiz
#feature göre parçalayabiliriz(yeni tablo oluşturabiliriz) value_vars= ['Attack','Defense']
#"frame" kısmına parçalamak istediğimiz data nın ismini veriyoruz 
# "id_vars" datayı hangi variable ye göre melt ediceğimiz (burda name göre yaptık)

# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Name', columns = 'variable',values='value')

# melt metodunu tekrar eski haline getirir. 
# Firstly lets create 2 data frame
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row

#concat metodu ile ilk veri son 5 veriyi birleştirdik
# axis 1 olmasının sebebi 2 datayı yukardan aşağı birleştirdik
#ignore_index=true olmasını sebebi ise ilk son datayı birleştirdiğimiz için yeni index numaraları attık
# çünkü son datalarını index en son da olduğu için
data1 = data['Attack'].head()
data2= data['Defense'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col

#data[['Attack',"Defense"]][0:5]  bu şekildede olur


int(0.8)   #0.8 int döüştürür 0 yuvarlar
#float("0.8") veya string bir değeri float çeviririz
data.dtypes # data daki veri tiplerini gösterir
# lets convert object(str) to categorical and int to float.
data['Type 1'] = data['Type 1'].astype('category') # object olan Type 1 feature sini tipini değiştirir category tipi yapar
data['Speed'] = data['Speed'].astype('float')      #  int olan Speed  feature sini değiştirir ve float tipi yapar

#data larda tür dönüşümü 
#örn mesala int girmesi olması geren bir data string olarak girilmiş hesaplama yapabilmek için tür dönüşümü yapılır
# As you can see Type 1 is converted from object to categorical
# And Speed ,s converted from int to float
data.dtypes
# Lets look at does pokemon data have nan value
# As you can see there are 800 entries. However Type 2 has 414 non-null object so it has 386 null object.
data.info()
# Lets chech Type 2
data["Type 2"].value_counts(dropna =False)
# As you can see, there are 386 NAN value

#dropna=false eğer nan değer varsa onuda göster demek
# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?

# "dropna"eğer type 2 feature sinde  null olan değer varsa onları çıkar 
# inplace = True ise çıkarttığımız data ları tekrar data1 kaydet 
#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true

# bu metot kontrol metodu diyebiliriz eğer 1==1 doğru hiç bişi döndürmez
# In order to run all code, we need to make this line comment
assert 1==2 # return error because it is false

# ama burda hata vericektir koşulu sağlmadığı için
assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values

# bu metot la listedeki null değerlerini ulaşırız ama null değerlerini sildiğimiz için hiçbişi döndürmedi
data["Type 2"].fillna('empty',inplace = True)

# type 2  yi empty ile dolduruyoruz
assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example
assert data.columns[1] == 'Name'        # assert kontrol işlemi yapabiliyorduk 1. colums= name dir eğer doğruysa hiçi bişi döndürmüyor yanlışsa hata veriyor
# assert data.Speed.dtypes == np.int     # Speed columns daki data ların type ları int mı lontrolunu yapıyoruz
# data frames from dictionary
country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df                             #dictionary den dataframe elde etme
                               # 2. yöntem ise csv formatında dataframe oluşturma
# Add new columns
df["capital"] = ["madrid","paris"]
df
# Broadcasting
df["income"] = 0 #Broadcasting entire column  # yeni bir column ekledik ve tüm değerlere sıfır atadık
df
# Plotting all data 
data1 = data.loc[:,["Attack","Defense","Speed"]]
data1.plot()
# it is confusing
# subplots
data1.plot(subplots = True) # her bi feature yı farklı grafikte göstermek için subplots kullanılır
plt.show()
# scatter plot  
data1.plot(kind = "scatter",x="Attack",y = "Defense")
plt.show()                           # 2 feature arasında ilişkiyi görmek için scatter kullanılır
# hist plot  
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)
# normed değerli 0-1 arasında normalize etmeye yarar
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
# cumulative parametresi frekansları toplaya toplaya gider

data.describe()
time_list = ["1992-03-08","1992-04-12"]    #liste 2 string değer var 
print(type(time_list[1])) # As you can see date is string  # listenin 1. indexnin tipini ekrana yazdırıyoruz
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)  # pandasın to_datetime liste nin türünü datetime çeviriyoruz
print(type(datetime_object))
# close warning
import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of pokemon data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)  # burda liste türünden datetime türüne dönüştürdük
data2["date"] = datetime_object              # ve data mıza yeni feature ekledik 
# lets make date as index
data2= data2.set_index("date")               # burdada datamızdaki mevcut indexleri değiştirdik 
data2                                        # artık zamana bağlı bir data oldu.
# Now we can select according to our date index
print(data2.loc["1993-03-16"]) # loc metodu datanın içinden belli bir endex sahip verileri çekmemize sağlar
                               # burda index artık tarih olduğu için o tarihteki veri çağırır
print(data2.loc["1992-03-10":"1993-03-16"])
# We will use data2 that we create at previous part
data2.resample("A").mean()    # burda resample("A") ile yıllara baz alıyoruz ve mean() o yıllardaki feature lerin ortlamalarını buluyoruz

# Lets resample with month
data2.resample("M").mean()  # burda aylara göre feature lerin ortalamalarını buluyoruz
# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate
# We can interpolete from first value
data2.resample("M").first().interpolate("linear")

# NAN değerleri linear bir şekilde doldurur 
# Or we can interpolate with mean()
data2.resample("M").mean().interpolate("linear") 
# aynı şekilde ortalamada boş olan değerleri linear bir şekilde dolduru  interpolate("linear")  metodu
# read data
data = pd.read_csv('../input/pokemon.csv')
data= data.set_index("#")  #varsayılan index 0 dan başlar bu metot ile data ların indexlerini 1 den başlatıyoruz
data.head()
# indexing using square brackets
data["HP"][1] # "HP" column daki birinici değeri yaz " dizi mantığı "
# using column attribute and row label
data.HP[1]
# using loc accessor
data.loc[1,["HP"]]  # 1.satır daki "HP" feature sindeki kesişen yer
# Selecting only some columns
data[["HP","Attack"]]
# Difference between selecting columns: series and dataframes
print(type(data["HP"]))     # series                         # dataframe ile series arasındaki fark
print(type(data[["HP"]]))   # data frames
# Slicing and indexing series
data.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive

# 1-10 kadar al aynı zamanda  "HP" - "Defenseye" kadar al
# Reverse slicing 
data.loc[10:1:-1,"HP":"Defense"]  # yukardaki sıralamayı tersten yazdırmak için yani 10-1 kadar
# From something to end
data.loc[1:10,"Speed":] #1-10 kadar "speed" - en sona kadar 
# Creating boolean series
boolean = data.HP > 200   # 200 den büyük olanları değişkene atar
data[boolean]             # burda true olanları yazdırır
# Combining filters
first_filter = data.HP > 150
second_filter = data.Speed > 35
data[first_filter & second_filter]
# Filtering column based others   # "HP" can feature si
data.HP[data.Speed<15]          # hızı 15 küçük olan verileri al ama bana canını göster
# Plain python functions       # NOT : apply metodu içine paramere olarak aldığı metodu çalıştırır
def div(n):
    return n/2  
data.HP.apply(div)  # data mı al canlarını al apply metodunu uygula
# Or we can use lambda function
data.HP.apply(lambda n : n/2)  # yukardaki işlem ile aynı işi görür
                               # n "HP" featurediki değişkenleri parametre olarak alır
# Defining column using other columns
data["total_power"] = data.Attack + data.Defense 
data.head()                              # yeni bir column oluşturduk ve değer olarak atatack ve defensenin toplamını verdik
# our index name is this:
print(data.index.name)
# lets change it
data.index.name = "index_name"
data.head()
# Overwrite index
# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to data3 then change index 
data3 = data.copy() 
# lets make index start from 100. It is not remarkable change but it is just example
data3.index = range(100,900,1) # yeni index 100 den başlasaın 900 kadar 1er 1er artsın
data3.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section
# It was like this
# data= data.set_index("#")
# also you can use 
# data.index = data["#"]   # data nın index istediğimiz feature yapabiliriz
# lets read data frame one more time to start from beginning
data = pd.read_csv('../input/pokemon.csv')
data.head()
# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index
data1 = data.set_index(["Type 1","Type 2"])  # artık 2 tane index oldu 
data1.head(100)
# data1.loc["Fire","Flying"] # howw to use indexes
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
# pivoting
df.pivot(index="treatment",columns = "gender",values="response")

# dataframe farklı bir açıdan bakmak için kullanılır 

df1 = df.set_index(["treatment","gender"])
df1
# lets unstack it
# level determines indexes
df1.unstack(level=0)  # 2 tane index varsa ilk index çıkartır
df1.unstack(level=1)  # 2 tane index varsa 2. index çıkartır
# change inner and outer level index position
df2 = df1.swaplevel(0,1)   
df2                   # ilk index ile 2. index in yerini değiştirmek için kullanılır
df
# df.pivot(index="treatment",columns = "gender",values="response")
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
# We will use df
df
# according to treatment take means of other features
df.groupby("treatment").mean()   # mean is aggregation / reduction method
# there are other methods like sum, std,max or min

# data yı "treatment" göre grubla ve mean metodu ile ortalamasını al
# we can only choose one of the feature
df.groupby("treatment").age.max() 

# "treatment" grubla sadace yaş colums al ve max() bul veya mean() ortalamısıda bulabiliriz
# Or we can choose multiple features
df.groupby("treatment")[["age","response"]].min() 

# "treatment" grubla "age" ve "response" için min leri bul
df.info()
# as you can see gender is object
# However if we use groupby, we can convert it categorical data. 
# Because categorical data uses less memory, speed up operations like groupby
#df["gender"] = df["gender"].astype("category")
#df["treatment"] = df["treatment"].astype("category")
#df.info()

