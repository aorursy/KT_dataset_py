# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool versi görselleştirme için



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon.csv')
data.info()
data.corr()#
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)#10 yerine () default olarak 5 deger verir veri setinden 10 yaparsan 10 deger verir
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')

data.Defense.plot(kind ='line', color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')

#yukarıda parametre için de yer alan line grafik te sagda yazan Speed karşılık gelir. legend o yazının konumunu

#ayarlar left sol right sagda upper yukarda lower aşagıda gibi

plt.xlabel('x axis')              # x eksenin de yazacak deger

plt.ylabel('y axis')              #y eksenin de yazacak deger

plt.title('Line Plot')            # plt.show kaldırırsan plotun türünu yazar plt.title

plt.show()
plt.scatter(data.Attack,data.Defense)
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')

plt.xlabel('Attack')              # label = name of label

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Speed.plot(kind = 'hist',bins = 50)

#plt.clf()#plottan sonra bunu kullanırsan plotu ortadan kaldırır.

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.values)

print(dictionary.keys)

print(dictionary)

del dictionary["spain"]

print(dictionary)

print('france' in dictionary) 

# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['spain'] = "barcelona"    # spain key degeri key e baglı olan madrid degerini "barcelona"

print(dictionary)

dictionary['france'] = "paris"       # dictionary['france'] = france bir key olmadıgı için yeni bir key ve value oluşturdu.

print(dictionary)

del dictionary['spain']              # keye sildigine göre tamamen ortadan kaybolor

print(dictionary)

print('france' in dictionary)        # dictionary içerisinde france diye bir key var mı diye bakıyor

print('vegas' in dictionary.values())

dictionary.clear()                   #dictionary tamamen siler.                

print(dictionary)

# In order to run all code you need to take comment this line

# del dictionary         # delete entire dictionary     

print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv('../input/pokemon.csv')

series = data['Defense']        # data['Defense'] = series buna vektör şeklinde uzanan denir

print(type(series))

data_frame = data[['Defense']]  # data[['Defense']] = data frame 

print(type(data_frame))

# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x = data['Defense']>200     # There are only 3 pokemons who have higher defense value than 200

data[x]
# 2 - Filtering pandas with logical_and

# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100

data[np.logical_and(data['Defense']>200, data['Attack']>100 )]
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

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['Attack']][0:1].iterrows():

    print(index," : ",value)



# example of what we learn above

def tuble_ex():

    """ return defined t tuble"""

    t = (1,2,3)#parantez içerisinde ise buna tuble denir.

    return t

a,b,c = tuble_ex()#Fonksiyon içerisinde 3 tane deger var ancak 2 tanesi kullanacaz diyelim a,b,_ yaparsan sadece ikisini kullanmış olursun.

print(a,b,c)
# guess print what

x = 2

def f():

    x = 3

    return x

print(x)      # x = 2 global scope

print(f())    # x = 3 local scope
# What if there is no local scope

x = 5#globol scope 

def f():

    y = 2*x        # there is no local scope x

    return y

print(f())         # it uses global scope x

# İlk önce fınksiyon içerisinde giriyor sonra fonk içinde x var mı diye bakıyor. x yok sa sonra global var mı diye bakıyor varsa işlemi yapar.

# How can we learn what is built in scope

import builtins

dir(builtins)

#Calıştırırsan daha önceden pythonda tanımlanmış olan ve belirli anlamlar taşıyor listeleri görürsen int,len vb
#nested function

def square():

    """ return square of value """

   

    def add():

        """ add two local scop variable """

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

print(f(5))#yukarıda b=1 c=2 default olarak tanımladıgımız için sadece 5 degerini gönderdik ve oldu

# what if we want to change default arguments

print(f(5,4,3))# b ve c değerini güncellemek istersem yeni değerler gönderirim
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)

# flexible arguments **kwargs that is dictionary

def f(**kwargs): 

    """ print key and value of dictionary"""

    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key, " ", value)

f(country = 'spain', capital = 'madrid', population = 123456)
# lambda function

square = lambda x: x**2     # where x is name of argument

print(square(4))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(1,2,3))
number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

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

print(z)

z_list = list(z)

print(z_list)
#Yukarıda birleştirdik burada tekrar ayırma işlemi yaptık

un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) # unzip returns tuble

print(un_list1)

print(un_list2)

print(type(un_list2))
# Example of list comprehension

num1 = [1,2,3]

num2 = [i + 1 for i in num1 ]

print(num2)
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
import pandas as pd

data=pd.read_csv("../input/pokemon.csv")

ort=sum(data.Speed)/len(data.Speed)

data["HIGH , LOW"]=["HIGH" if ort >i else "LOW" for i in data.Speed]

data.loc[:10,["HIGH , LOW","Speed"]]
# lets return pokemon csv and make one more list comprehension example

# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

threshold = sum(data.Speed)/len(data.Speed)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]] # we will learn loc more detailed later
import pandas as pd

data=pd.read_csv("../input/pokemon.csv")

data.head(5)

ort=sum(data.Speed)/len(data.Speed)

print(ort)

data["HIGH OR LOW"] =["high" if i>ort else "low" for i in data.Speed]

data.loc[:10,["HIGH OR LOW","Speed"]]
import pandas as pd

data = pd.read_csv('../input/pokemon.csv')

data.head()  # ilk 5 tanesini tablo şeklinde gösterdi
# tail shows last 5 rows

data.tail() #son 5 tanesini tablo şeklinde gösterdi
# columns gives column names of features

data.columns
# shape gives number of rows and columns in a tuble

data.shape

#(800,12) çıkar tabloda 800 tane pokemon oldugu 12 tane de özellik oldugu 12 tane sütün da diyebiliriz.
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
import pandas as pd

data=pd.read_csv("../input/pokemon.csv")

data.head(2)
# value_counts öğrendik

print(data['Type 1'].value_counts(dropna =False))

#Water pokemonlar içerisinde 1112 tane water var diyor

#Normal 98 tane bundan var oldugunu söylüyor.
# For example max HP is 255 or min defense is 5

data.describe() #ignore null entries

#lower quartile=%25 O1 upper quartile=%70 Q3 median(medyan=%50 Q2) diye geçer yazdıgım gibi oku direk

#count=Hp canların toplamı/Defense degerlerin toplamı gibi

#min=hp en az canı olan max=en fazla canı olan

#mean=degerlerin ortalaması 1,1,1,1,100 bu yüzden lower-upper quartile kullanılır. İşçi Maaş Örnegi

#Mean mantıgı şı patron geldi ort maaş kaçtı dedi zam yapacagım dedi ne yaparsın 1+1+1+100 toplarsın 

#4 e bölersin ama oldu mu mu haksızlık var oysaki median baksak ortadaki degere alt ve üst sınır degerine

#bakar ona göre maaş ekler

#17 VİDEO
import matplotlib.pyplot as plt

# For example: compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

data.boxplot(column='Attack',by = 'Legendary')

plt.show()

#şimdi tablo değerleri çıkacak aşagıda yukarlak ile gösterilen bize diyorku Attack degerlerinde aykırı çok fazla deger var diyor

#yuvarlak aşagıda olsa aykırı çok az degerler var diyecekti out layer denir bu işleme fazla olmasına
# Firstly I create new data from pokemons data to explain melt nore easily.

data_new = data.head()    # Tablo daki 5 dgeri al bunuda data_new içine at bunu melt() edecez o aşagıda 

data_new
# lets melt

# id_vars = Sabit kalacak yer 

# value_vars = what we want to melt

#id_vars='Name isimler yukarı daki tablo ile aynı kalacak her hangi bir eğişiklik yapılmayacak anlamına gelir.'

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Speed','Defense'])

melted

#Tablodaki belirli değeleri incelememizi saglar. İlerde görselleştirme için kullanacaz bunu öğren
# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'Name', columns = 'variable',values='value')

#yukarı da yaptıgımız melted eski haline getirdi

# Firstly lets create 2 data frame

#Seçilen dataları birleştirme

#axis=0 yukarıdan aşagıaya dogru birlşeitme için =0

#ignore_index =True yeni index ata son 5 kayıt geldi 700 index var ama bunu index atayarak 4 yaptık tail() çalıştır bak

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
#Yukarıdaki dikey bir şeklindeydi şimdi yatay şeklinde sadece istediğimiz yerleri alıp birleştirme

#axis =1 yanyana birleştir burda kullanılır.

data1 = data['Attack'].head()

data2= data['Defense'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes

#object yazan yer string demek buarada
# lets convert object(str) to categorical and int to float.

data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')#degerleri floata çevirdi
# As you can see Type 1 is converted from object to categorical

# And Speed ,s converted from int to float

data.dtypes
# Lets look at does pokemon data have nan value

# As you can see there are 800 entries. However Type 2 has 414 non-null object so it has 386 null object.

data.info()

# Lets chech Type 2

data["Type 2"].value_counts(dropna =False)

# kaç tane kedi köpek varsa onları bul köpek ten 10 tane kediden 5 tane gibi

#bu kodu çalıştır hangi Type 2 hangi pokemondan kaç tane oldugunu karşısında yazar ancak dropna= false bu bize type 2 kaç tane nan boş oldugunu gösterir.
# Lets drop nan values

data1=data   # also we will use data to fill missing value so I assign it to data1 variable

data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

# Type 2 si olmayan pokemonları listeden at çıkar demek nan ları çıkartıyor

#Çalıştırdıgında her hangi bir şey olmayacak ama aslında oldu olup olmadıgını kontrol etmek için aşıgıdan devam ediyoruz
#  Lets check with assert statement

# Assert statement:

assert 1==1 # bu kodu çalıştırırsam hiç bir şey yapmaz çünkü 1==1 e eşit
# In order to run all code, we need to make this line comment

#assert 1==2 # bu kodu çalıştırırsam hata verir çünkü 1==2e eşit değil
assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values

#şimdi o yukardaki listeden attı mı atmadı mı olayını kontrol edecez

#hiç bir geri dönüş yapmadıgına göre biz type 2 de nan ların hepsini listeden attık demektir. Sonra
data["Type 2"].fillna('empty',inplace = True)

#tam anlatmadı
assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values
data.head()
#data.Speed.dtypes tipine baktık
# bu kod data.head() tablo columns ları dogru mu diye kontrol ediliebilir 4 columns 'Hp' var mı varsa herhangi bir şey döndürmez yoksa hata verir..

assert data.columns[4] == 'HP'

assert data.Speed.dtypes == np.int# bu satır ise data.Speed degerlerin int olup olmadıgına bakar dogru ise hiç bir şey döndürmez
# EN TEMELDEN CSV BENZERİ DATAFRAME OLUŞTURMAK

country = ["Spain","France"]

population = ["11","12"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
## Add new columns yukarıdaki tabloya yeni columns ekleme

df["capital"] = ["madrid","paris"]

df
# Broadcasting

df["income"] = 0 #yeni bir columns oluşturdu ve hepsine 0 degerini atadı

df
# Plotting all data 

data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()

# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="Attack",y = "Defense")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)

#range(0,250) x eksenin 0 dan 250 ye kadar olması tek farkı

#normed=datamızı normalize etmesi 0 la 1 arası yapması
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt

#cumulative olasıkları toplaya toplaya gider
data.describe()#quarlite olayı işte
data.head()
#datamızda time olmadıgı için time listesi oluşturduk

time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 

#Dataya indexin yanına tarihli index ekledi zamana baglı data yaptık
print(data.loc[0]) #direk data içerisindeki 0 numaralı kaydı getirir.
# Now we can select according to our date index

print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
# A yıla göre resample et yani 1992 ve 1993 var bu yılların tüm degerlerinin ort al

#92 yılında 3 tane pokemın var onların can ortlamsını buluyor

data2.resample("A").mean()
data2.head()
# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
# read data

data = pd.read_csv('../input/pokemon.csv')

data= data.set_index("#")

data.head()

#Eski data.head() çalıştır birde bunu açlıştır 2 tane index vardı # biride 1234 diye giden artık tek şekil haline geldi 2. satır sayesinde
# yukarda indexi ayarladık ya ayarladıgımız için artık 1. sırada ayarlamasak pandas da 0. sırada yer alır.

data["HP"][1]#data içerisinde can olan ve 1.index te yeralan can degeri
# yukarı ile aynı

data.HP[1]
# datanın 1 satırı can sütünündaki deger

data.loc[1,["HP"]]
# Selecting only some columns

data[["HP","Attack"]]

#data daki can ve atak kısımlalrını yazdırır. 800 tane hepsini ver
# Difference between selecting columns: series and dataframes

print(type(data["HP"]))     # series

print(type(data[["HP"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive

#1 den 10 kadar can dan defansa kadar alır yani aradaki attack yazdırır.
# Reverse slicing 

data.loc[10:1:-1,"HP":"Defense"]  #10 dan geriye dogru yukarı ile aynı
# From something to end

data.loc[1:10,"Speed":] #1 satırdan 10 e kadar Speeden başla en sona kadar al
# Creating boolean series

boolean = data.HP > 200

data[boolean]

#false yazdırmaz 200 den büyük olanları yazdırır.
# Combining filters

first_filter = data.HP > 150

second_filter = data.Speed > 35

data[first_filter & second_filter]

#&and ve anlamında
data[data.Speed<15]#Hızı 15 den küçük olanlar
# Filtering column based others

data.HP[data.Speed<15]#hızı 15 den küçük olanları seçti sonrada onların can degerlerini ekrana bastı
# Plain python functions

def div(n):

    return n/2

data.HP.apply(div)#fonk siyon oluşturmuş fonksiyonda 2 ye bölme data ki da can ları fonksiyona attı ve can degerlerini iki ye böldü oda ekrana yazdı
# Or we can use lambda function

data.HP.apply(lambda n : n/2)#lambda ile de yapılaiblir yukarıdaki işlem
# Defining column using other columns

data["total_power"] = data.Attack + data.Defense

data.head()

#attack ve defans toplamı olsun data.Attack + data.Defense bu ikisini toplayıp yeni bir columns dataya ekledi ve degerleri yazıdrdı
# our index name is this:

print(data.index.name)#index in nameine bakıyoruz index adı adı ne 

# lets change it

data.index.name = "index_name" #index in adı # biz bunu index_name yaptık

data.head()
# Overwrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data3 then change index 

data3 = data.copy()

#yukardaki 2 satıra takılma data index bozulmasın diye öyle yaptık aslında orda da bir datanın nasıl kopyalanacagı hakkında bilgi aldık

data3.index = range(100,900,1)# data index i 1 den başlıyordu ya biz bunu 100 den 900 kadar 1 den art şeklinde tanımladık

data3.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section

# It was like this

# data= data.set_index("#")

# also you can use 

#data.index = data["#"]
# lets read data frame one more time to start from beginning

data = pd.read_csv('../input/pokemon.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# #çok faydalı bu kod yani yukarıdaki tabloya göre anlatacam head e göre type1 de yer alan grass index oluyor type1 karşı gelenleri yazdıyor aslında şöyle

#grass çıkacak type1 de sonra type2 ye baktıgımızda grass olanların pokemon isimleri gibi 

#şöyle anlatıyım type1 mavi göz olsun o zaman mavi gözlü kedi isimleri yaşları cinsleri gibi olacak tablo

data1 = data.set_index(["Type 1","Type 2"]) 

data1.head(100)

# data1.loc["Fire","Flying"] # howw to use indexes
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df

#tablo oluşturma gibi düşün çalıştır bak 
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1

# lets unstack it
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])#bunu görmüştük

#çalıştır bak
# We will use df

df
# according to treatment take means of other features

df.groupby("treatment").mean()   # mean is aggregation / reduction method

# treatment iki tane tedavi var A VE B bunlara göre gurpla ve ort sını al response toplayacak ort alacak yazdırıacak aynı şekilde age içinde yapacak
# we can only choose one of the feature

df.groupby("treatment").age.max() #A Ve B vardı ya grupladı sonra yaşının en büyük degeri neyse onu yazdır demek
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min() #min degerini yazdırır
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby


