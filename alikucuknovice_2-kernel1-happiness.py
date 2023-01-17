# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/world-happiness/2019.csv')
data.info()
data.head()
data.corr()   #GDP per capita ve Generosity neredeyse ters orantı
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)                    #ÖNEMSİZ

plt.show()
data.columns
data.rename(columns={'Overall rank': 'Overall_rank'}, inplace=True)

data.rename(columns={'Country or region': 'Country_region'}, inplace=True)

data.rename(columns={'GDP per capita': 'GDP'}, inplace=True)

data.rename(columns={'Social support': 'Social_support'}, inplace=True)

data.rename(columns={'Freedom to make life choices': 'Freedom'}, inplace=True)

data.rename(columns={'Perceptions of corruption': 'Perceptions_of_corruption'}, inplace=True)

data.rename(columns={'Healthy life expectancy': 'Healthy_lifeexp'}, inplace=True)

data.columns
data.Social_support.plot(color = 'g',label = 'Social_support',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.GDP.plot(color = 'r',label = 'GDP',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')               #x axis index yani 156 tane.

plt.show()
data.plot(kind='scatter', x='Score', y='Social_support',alpha = 0.5,color = 'red')   #alpha, saydamlık

plt.xlabel('Score')              # label = name of label

plt.ylabel('Social_support')

plt.title('Score Social_support Scatter Plot')            # title = title of plot

plt.show
data.Score.plot(kind = 'hist',bins = 50,figsize = (12,12))  

plt.show()
data
x= data["Score"]>5

x                    # x yazarsan sadece true false veriri data içinde matris verir. ÇÜNKÜ TYPE DATAFRAME DİR. 97 TANE.

data[x]
x=data[(data['Score']>7) & (data['Social_support']>1)]



x
print("hello")
list1 = [3,5,7]

y = map(lambda x: x**2,list1)

print(list(y))
list1 = [5,10,15]

list2 = [i**2 if i == 10 else i-5 if i<7 else i + 5 for i in list1]

print(list2)
dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())







dictionary['spain'] = "barcelona"    # update existing entry

print(dictionary)





dictionary['france'] = "paris"       # Add new entry

print(dictionary)





del dictionary['spain']              # remove entry with key 'spain'

print(dictionary)





print('france' in dictionary)        # check include or not





dictionary.clear()                   # remove all entries in dict

print(dictionary)
# default arguments

def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print(f(5))









# what if we want to change default arguments

print(f(5,4,3))









def f(*args):

    for i in args:

        print(i)

f(1)
lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

    

    



for index, value in enumerate(lis):

    print(index," : ",value)

    

    

    

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

    

    

    





for index,value in data[['Score']][0:1].iterrows():

    print(index," : ",value)

    

    

    

    

    i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
x = 2

def f():

    x = 3

    return x

print(x)      # x = 2 global scope

print(f())    # x = 3 local scope
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
def f(*args):

    for i in args:

        print(i)

f(1)









square = lambda x: x**2     # where x is name of argument

print(square(4))







tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(1,2,3))







number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

print(list(y))









list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

z_list = list(z)





print(z_list)
num1 = [1,2,3]

num2 = [i + 1 for i in num1 ]

print(num2)











num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)





threshold = sum(data.Score)/len(data.Score)

threshold

data["ortalamaskordan"] = ["high" if i>threshold else "low" for i in data.Score]





data.loc[:, ["ortalamaskordan","Score"]]

data.loc[::-1, ["ortalamaskordan","Score"]]

# arada boşluk olduğu için kodu çağıramayız Ör; Type 1 bu yüzden data["Type 1"] yaparız
data.tail()
data.shape                #156 satır ,10 sütun(future) var
data.columns
data.describe()
data.boxplot(column='Score')    

# line at top is max

# line at top is 75%

# line is median (50%)

# line at bottom is 25%

# line at bottom is min
data_new= data.head()

data_new



melted_ornegi = pd.melt(frame=data_new,id_vars = 'Country_region', value_vars= ['Score','Generosity'])

melted_ornegi





# melted_ornegi eski haline getirelim



melted_ornegi.pivot(index = 'Country_region', columns = 'variable',values='value')

data1 = data[0:9]

data2= data[0:3]





conc_orn = pd.concat([data1,data2],axis =0,ignore_index =True)                         # axis = 0 : adds dataframes in row,yani vertical birleştirir daha mantıklı





conc_orn
data1 = data['GDP'].head(8)

data2= data['Social_support'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
int(0.8)
data.dtypes
data['Country_region'] = data['Country_region'].astype('category')

data['Score'] = data['Score'].astype('int')



data                                                               #örneğin int leri string yazmışlar böylece inte çevrriz yararlı. ama şimdi eski haline döndüremedim

                                                                    #o yüzden data setimi baştan okutalım.
data.loc[:1,"Score"] = 'NAN'

data.loc[:1,"Score"]

data = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

data.rename(columns={'Overall rank': 'Overall_rank'}, inplace=True)                              #verileri en baştaki hale getirmek için.

data.rename(columns={'Country or region': 'Country_region'}, inplace=True)

data.rename(columns={'GDP per capita': 'GDP'}, inplace=True)

data.rename(columns={'Social support': 'Social_support'}, inplace=True)

data.rename(columns={'Freedom to make life choices': 'Freedom'}, inplace=True)

data.rename(columns={'Perceptions of corruption': 'Perceptions_of_corruption'}, inplace=True)

data.rename(columns={'Healthy life expectancy': 'Healthy_lifeexp'}, inplace=True)

data.columns

data
data["Country_region"].value_counts(dropna =False)         # dropna false boş olanlarıda göster demek, hangi stringden ne kadar var gösterir.
data["Type 2"].value_counts(dropna =False)          #pokemon type2 de boş veriler vardı



data1=data



data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data



assert  data['Type 2'].notnull().all()   #bunu çalıştırdığında hiçbirşey dönmez,çünkü artık boş data yok,kntrol yapılır





#yada



data["Type 2"].fillna('empty',inplace = True)    # empty diye dolduruyoruz



assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values,empty yaptık

assert data.columns[1] == 'Score'         # assert ile kontrol yapılır, bunu çalıştırdığınd ahata verir Country_region bu yazsaydı score yerine birşey dönmezdi.
# data frames from dictionary from list





country = ["Spain","France"]

population = ["11","12"]



list_label = ["country","population"]



list_col = [country,population]





zipped = list(zip(list_label,list_col))

zipped #series verir burada çalıştırırsan



data_dict = dict(zipped)



df = pd.DataFrame(data_dict)



df





# Add new columns

df["capital"] = ["madrid","paris"]

df



# Broadcasting

df["income"] = 0 #Broadcasting entire column

df

data.columns
data1 = data.loc[:,["Score","GDP","Social_support"]]

data1.plot()

# it is confusing
# subplots

data1.plot(subplots = True)

plt.show()
data1.plot(kind = "hist",y = "Score",bins = 50,range= (0,250),normed = True)    #normed ve range yeni
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) 



# As you can see date is string

# however we want it to be datetime object





datetime_object = pd.to_datetime(time_list)



print(type(datetime_object))

datetime_object



# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head()





date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]





datetime_object = pd.to_datetime(date_list)





data2["date"] = datetime_object





# lets make date as index

data2= data2.set_index("date")





data2 
#index, tarih oldu, tarihe göre yazdırabiliriz.

print(data2.loc["1993-03-16"])





print(data2.loc["1992-03-10":"1993-03-16"])
#Resampling: statistical method over different time intervals

#Needs string to specify frequency like "M" = month or "A" = year







data2.resample("A").mean()

data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")           #nümeric olanları lineer dolduruyor.
#or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")       #meanleri bozmucak şekilde doldurur.
data = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

data.rename(columns={'Overall rank': 'Overall_rank'}, inplace=True)                              #verileri en baştaki hale getirmek için.

data.rename(columns={'Country or region': 'Country_region'}, inplace=True)

data.rename(columns={'GDP per capita': 'GDP'}, inplace=True)

data.rename(columns={'Social support': 'Social_support'}, inplace=True)

data.rename(columns={'Freedom to make life choices': 'Freedom'}, inplace=True)

data.rename(columns={'Perceptions of corruption': 'Perceptions_of_corruption'}, inplace=True)

data.rename(columns={'Healthy life expectancy': 'Healthy_lifeexp'}, inplace=True)

data.columns

data
data["Score"][1]

data.Score[1]

data.loc[1,["Score"]]

data[["Score","GDP"]]
# Difference between selecting columns: series and dataframes

#print(type(data["HP"]))     # series

#print(type(data[["HP"]]))   # data frames
#data.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive

#data.loc[10:1:-1,"HP":"Defense"] 

#data.loc[1:10,"Speed":] 
boolean = data.Score > 5                 #boolean, true , false demek

data[boolean]
first_filter = data.Score > 5

second_filter = data.GDP > 1

data[first_filter & second_filter]
data.GDP[data.Score<7]
def div(n):

    return n/2

data.Score.apply(div)





#or



data.Score.apply(lambda n : n/2)   #better
data["total_power"] = data.Score + data.GDP

data.head()
print(data.index.name)

data.index.name = "index_name"

data.head()





#data= data.set_index("index_name")         # BÖYLECE İNDEX LER SIFIRDAN DEĞİL 1 DEN BAŞLAR

#data.head()
data.index = range(1,157,1)

data.head()



print(data.index.name)

data.index.name = "index_name"

data.head()

data
#istersek indexi bir sütunada eşitleyebiliriz,mesala overall rank yapaılm



data.index = data["Overall_rank"]

data.head()
data = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

data.rename(columns={'Overall rank': 'Overall_rank'}, inplace=True)                              #verileri en baştaki hale getirmek için.

data.rename(columns={'Country or region': 'Country_region'}, inplace=True)

data.rename(columns={'GDP per capita': 'GDP'}, inplace=True)

data.rename(columns={'Social support': 'Social_support'}, inplace=True)

data.rename(columns={'Freedom to make life choices': 'Freedom'}, inplace=True)

data.rename(columns={'Perceptions of corruption': 'Perceptions_of_corruption'}, inplace=True)

data.rename(columns={'Healthy life expectancy': 'Healthy_lifeexp'}, inplace=True)

data.columns

data
data["bolge"] = data.Score<4

data.head(156)
data1 = data.set_index(["bolge","Score"]) 

data1.head(156)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df







#şimdi bu iki tedavinin genderlarına göre responselarını görmek istiyorum,pivot et.



df.pivot(index="treatment",columns = "gender",values="response")





#outer,inner sıralama



df1 = df.set_index(["treatment","gender"])

df1



#indexlerin yerini değiştirmek



df2 = df1.swaplevel(0,1)

df2





#melt



pd.melt(df,id_vars="treatment",value_vars=["age","response"])


df.groupby("treatment") .mean()

df.groupby("treatment").age.max() 
df.groupby("treatment")[["age","response"]].min() 
#markdownda renk için < font color = 'blue' >

# linklerde köprü için



#   [Load and Check Data](#1)



# ve alttakine de bunu yaz , <a id = "1"></a><br>
a = [1,2,3,4]

plt.plot(a)

plt.show()