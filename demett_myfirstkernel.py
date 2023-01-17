# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
data.head()
data.info() 
data.corr()
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,linewidths=.4, fmt='.2f',ax=ax)



#f,ax=plt.subplots(figsize=(15,15))

#sns.heatmap(data.corr(),annot=True,linewidths=.5,cmap="YlGnBu", fmt='.1f',ax=ax)
data.columns
#line plot

plt.plot(data.year,data.population, color="red", label="suicide")

#data.year is x label,data.population is y label

#if one label not specified, x label will be index of the data frame automatically. 



plt.xlabel("year")

plt.ylabel("population")

plt.title("year-population")

plt.legend()



plt.show
#another line plot

data.year.plot(kind = 'line', color = 'g',label = 'year',linewidth=1,alpha = 0.8,grid = True,linestyle = ':')

#alpha =opacity

plt.legend(loc='upper right') #loc= location

plt.ylabel('years')

plt.xlabel('index')

plt.title('Line Plot-year') #figure title

plt.show()

#if you want, you can add different features with different color lines or linestyles
#scatter plot

data.plot(kind='scatter', x='year', y='population',alpha = 0.4,color = 'red')

#also you can do like this, it's the same

#plt.scatter(data.year,data.population,color="red",label="scatter",alpha = 0.4)

plt.xlabel('year')

plt.ylabel('population')

plt.title('Scatter Plot')

plt.show()
#histogram

data.population.plot(kind='hist',bins=30,figsize=(5,5),color='black')

#bins= number of bar in figure

#or you can do like this

#plt.hist(data.population, bins=30,color='black')

plt.show()
#subplot

plt.subplot(2,1,1)

plt.plot(data.year, color="red",label="year")

plt.ylabel("year")

# there are 2 columns. Second row of first column

plt.subplot(2,1,2) 

plt.plot(data.population,color="green",label="population")

plt.ylabel("population")
plt.clf() #cleans the plot.
data.head()
series=data['suicides_no']

dataFrame= data[['suicides_no']]



print("dataFrame" ,type(dataFrame))



print("series" ,type(series))
print(dataFrame)

print("--------")

print(series)
data[(data['suicides_no']>9000) & (data['year']>2010 )]

#show suicides_no greater than 9000 and  year greater than 2010 from the data



#y=data['year']>2010

#s=data['suicides_no']>9000

#data[s&y]
print(data['suicides_no'][0])

print(data['suicides_no'][1])

print(data['suicides_no'][0]>=data['suicides_no'][1])
data[(data['sex']=='male') & (data['year']>=2015 )]
albaniaCount=0 # albania count in data

others=0 #other countries count in data

for i in data['country']:

    if i=='Albania':

        albaniaCount+=1

    else:

        others+=1

print("Ratio of Albania to other countries= ",albaniaCount/others)
for index,value in data[['generation']][-5:].iterrows():

    print(index," : ",value) 

    #last 5 values and index
data['country'].unique() #all different countries
dic={} #create empty dictionary

for each in data['country'].unique(): 

    dic[each]=0 #all different countries are our keys and all values are zero



for i in data['country']:

    dic[i]+=1 #return all country data, increase own value for each country.



for key,value in dic.items():

    print(key,":",value) #all keys and values



#print(dic)    
lis=[]

for each in data['generation'].unique(): #different generations

    lis.append(each) # add this generations to list

    

for index, value in enumerate(lis):

    print(index,":",value)

#built in scope

import builtins 

dir(builtins)
sex="male" #global scope



def female():

    """show sex variable"""

    sex="female" #local scope

    print(sex)

if data.sex[2] =="female":

    female()

print("...")

print(sex)
sex="female" #global scope



def female(): #there is no local scope

    """show sex variable"""

    print(sex)

if data.sex[2] =="female":

    female()
def albania(): #first function

    """return ratio of suicide numbers to years of Albania"""

    def albaniaSuicide(): #second function inside first func.

        """return sum of suicide numbers in data"""

        summ=0

        for index, value in enumerate(data["country"]):

            if value=="Albania": #if the country is Albania

                summ+=data.suicides_no[index] #then add the number of suicides in that index

        return summ 

    year=len(data['year'].unique()) #1985-2016

    return albaniaSuicide()/year

        

albania()  
#default arguments

def sexRatio(a=3): #a=3 default argument, a is number of digits after comma

    """show the suicide ratio of female to male in the world"""

    dictionary={"male":0,"female":0}

    for index,value in enumerate(data["sex"]):

        dictionary[value]+=data["suicides_no"][index]

    print("ratio of female to male: " ,round(dictionary["female"]/dictionary["male"],a))

    

sexRatio() # used default argument(a=3)

sexRatio(1) # used a=1
def show(country,*args): #flexible argument

    """print some details about suicide"""

    print("country:",country)

    print("other details: ")

    for i in args:

        print(i)

for i in [1,2,3]:   

    show(data["country"][i],data["age"][i],data["population"][i])

#different number of parameters

for i in [1,2,3]:   

    show(data["country"][i],data["age"][i])    
def x(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items():            

        print(key, ":", value,".")

for i in [1,2,3]:   

    x(year = data["year"][i], suicides_no = data["suicides_no"][i])
rate= lambda x,y: x/y #x,y are names of arguments

print(rate(data["suicides_no"][5],data["population"][5]))
y = map(lambda x:round(x,-1),data.suicides_no)

print(list(y)[:15]) #first 15 items
years=[2000,2001,2002,2003]

suicides_no=[]



for each in years:

    s=0

    x=data['year']==each

    s=sum(data[x].suicides_no) #sum all suicide numbers each year

    suicides_no.append(s) # add list  

#zip        

z=zip(years,suicides_no) 

print(type(z))

list_z=(list(z)) #convert list

print(list_z)

print(".....")



#unzip

un_z=zip(*list_z)

un_1,un_2=list(un_z)

print(un_1)

print(un_2)
x=data.suicides_no

y=["too much" if i>50 else "much" if i>10 else "normal" for i in x]

print(y[:50])