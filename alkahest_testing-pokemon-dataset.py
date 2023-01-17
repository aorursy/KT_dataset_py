# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#We will use pandas for importing data 
pokedata = pd.read_csv('../input/pokemon/Pokemon.csv')
countrydata = pd.read_csv('../input/countries/countries.csv')
pokedata.info()
#pokedata.columns
#pokedata.describe()
#pokedata.corr()
#pokedata.head()
f,ax = plt.subplots(figsize=(9, 9))
sns.heatmap(pokedata.corr(), annot=True, linewidths=0.1, fmt= '.1f',ax=ax)
plt.show()
pokedata.HP.plot(kind='line', grid=True, label='Hp', color='r', alpha=0.6, linestyle='-.', linewidth=1)
pokedata.Defense.plot(label='Defense', color='g', alpha=0.6, linestyle=':', linewidth=1)
plt.legend()
plt.show()
pokedata.Generation.plot(kind='hist', grid=True, bins=20)
plt.show()
# Hangi jenerasyondan kaç tane pokemon var
# Bu veriye göre attack arttıkca sp atk artıyor diyebilirz
pokedata.plot(kind = 'scatter', x = 'Attack', y='Sp. Atk', grid = True, alpha = 0.5, color = "#4700ab")
plt.title("Pokemons's Attack and Special Attack")
plt.show()
countrydata.country.unique() # with this method we can see unique values of country
#countrydata['country'].value_counts() # Anotheher way to find unique values (and counts)
turkey = countrydata[countrydata.country == "Turkey"]
germany = countrydata[countrydata.country == "Germany"]
turkey.head(12)
#germany.head(12)
turkey.plot(kind='line', x='year', y='population', grid=True)
plt.ylabel("Bilion")
plt.show()
# Turkey's population 3x than Germany in 50 years
plt.plot(turkey.year, turkey.population / turkey.population.iloc[0], label='Turkey') #turkey.population.iloc[0] = turkeys population index 0
plt.plot(germany.year, germany.population / germany.population.iloc[0], label='Germany')
plt.legend()
plt.xlabel('year')
plt.ylabel('population growth multiple')
plt.show()
def growth(country):
    for i in range(len(country)):# 0,1,2,3....11 (12)
        length = len(country)-1
        if length>i:
            x = round(((country.iloc[i+1]-country.iloc[i])/country.iloc[i])*100,2) # virgülden sonraki basamak sayısını belirleme
            print("geçen yıla göre nüfusun yüzdelik artış/azalışı: {}".format(x))
        else:
            print("population growth 1952-2007 = %{}".format(int((country.iloc[length]/country.iloc[0])*100)))
growth(turkey.population)
#growth(germany.population)
turkey["growth"] = [2.54,2.54,2.54,2.54,2.54,2.54,2.54,2.54,2.54,2.54,2.54,0]
turkey.head(12)
def growth(country):
    for i in range(len(country)):# 0,1,2,3....11 (12)
        length = len(country)-1
        if length>i:
            x = round(((country.iloc[i+1]-country.iloc[i])/country.iloc[i])*100,2) # virgülden sonraki basamak sayısını belirleme
            turkey.growth[1572+i] = x # 1572 first index
            print(x)
        else:
            g = round((country.iloc[length]/country.iloc[0]),2)
            turkey["tgrowth"] = [g,g,g,g,g,g,g,g,g,g,g,g]
print(growth(turkey.population))
turkey.head(12)
germany["growth"] = [2.54,2.54,2.54,2.54,2.54,2.54,2.54,2.54,2.54,2.54,2.54,0]
germany.head(12)
def growth(country):
    for i in range(len(country)):# 0,1,2,3....11 (12)
        length = len(country)-1
        if length>i:
            x = round(((country.iloc[i+1]-country.iloc[i])/country.iloc[i])*100,2) # virgülden sonraki basamak sayısını belirleme
            germany.growth[564+i] = x # 1572 first index
            print(x)
        else:
            g = round((country.iloc[length]/country.iloc[0]),2)
            germany["tgrowth"] = [g,g,g,g,g,g,g,g,g,g,g,g]
print(growth(germany.population))
germany.head(12)
turkey.head(12)
turger = pd.concat([turkey,germany], axis=0, ignore_index=True) # axis=0 vertical 1 horizontal
turger
#pokedata.Attack.mean() -- > mean of Attack

#SHORT VERSION WITH LIST COMP
pokedata["meanAttack"] = ["Low" if i<pokedata.Attack.mean() else "High" for i in pokedata.Attack]
pokedata.head()

#LONG VERSION WITH FOR LOOP
#for i in pokedata.Attack:
#    if i<pokedata.Attack.mean():
#        ..."Low"
#    else:
#        ..."High"
# WHAT IS SPLIT?
print("[0]= ",pokedata.columns[0].split())
print("[1]= ",pokedata.columns[1].split())
print("[2]= ",pokedata.columns[2].split())
print("[2,0]= ",pokedata.columns[2].split()[0])# 0 index = 'Type'
print("[2,1]= ",pokedata.columns[2].split()[1]) # 1 index = '1'
pokedata.columns
pokedata.columns = [i.split()[0]+"_"+i.split()[1] if(len(i.split())>1) else i for i in pokedata.columns]
pokedata.columns
pokedata.columns = [i.lower() for i in pokedata.columns]
pokedata.columns
pokedata = pd.read_csv('../input/pokemon/Pokemon.csv') # CHECKPOINT
pokedata.Attack.describe()
pokedata.boxplot(column='Attack',by='Legendary')
plt.show()
newData = pokedata.head()
newData
# id_vars = what we do not wish to melt (id )
# value_vars = what we want to melt
# frame = Data Frame
# vars = variables
melted = pd.melt(frame=newData,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted
# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Name', columns = 'variable',values='value')
data1 = pokedata.head()
data2 = pokedata.tail()
verConcat = pd.concat([data1,data2], axis=0, ignore_index=True)
# ignore_index
#0 1 2 3 4 5 6 7 8 9, True
#0 1 2 3 4 795 796 797 798 799, False
#data2
verConcat
data1 = pokedata['Attack'].head()
data2= pokedata['Defense'].head()
horConcat = pd.concat([data1,data2],axis =1)
horConcat
pokedata.dtypes
pokedata['Type 1'] = pokedata['Type 1'].astype('category')
pokedata['Attack'] = pokedata['Attack'].astype('float')
pokedata.dtypes
pokedata.info()
# We can see Type 2 have 386 null
# Lets chech Type 2
pokedata["Type 2"].value_counts(dropna =False) # we are using dropna=False here to see NaN values
pdata = pokedata.copy()
pdata["Type 2"].dropna(inplace = True) # drop the NaN values
pdata["Type 2"].value_counts(dropna =False) # Another way to check is 'assert'
# Now we can't see any NaN values
assert 1==1 # if return false ERROR!!
assert 1==2 # 1 not equal 2 so gives ERROR
assert pdata['Type 2'].notnull().all() # gives nothing because we droped the null values
pdata["Type 2"].fillna('empty',inplace = True)
assert pdata['Type 2'].notnull().all() # gives nothing because we have no null values
Champion = ['Irelia', 'Aatrox', 'Gankplank', 'Jax'] # values of Champion
Attack = ['63', '60', '64','70'] # values of Attack
label = ['Champion', 'Attack'] # column names
column = [Champion, Attack]
zipped = list(zip(label, column))
#zipped
data_dict = dict(zipped) # for convert to dataframe first we have to convert dictionary
#data_dict
dataFrame = pd.DataFrame(data_dict)
dataFrame
print(dataFrame.columns)
print(dataFrame.values)
dataFrame['Armor'] = ['34', '33', '35', '36']
dataFrame['Magic_resist'] =['32', '36', '32', '33'] # real magic resists is 32
dataFrame
dataFrame['Movement_speed'] = 325
dataFrame
data1 = pokedata.loc[:,["Attack","Defense","Speed"]]
#data1.plot() # bad
data1.plot(subplots=True, figsize=(7,7)) # good
plt.show()
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)# normed = about y axis 
#data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = False)
plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)# 2x1 2 column # self run this 3x2 3 column 2 feature total 6
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.show()
def div(n):
    return n/2
data1.HP.apply(div)
data = pd.read_csv("../input/pokemon/Pokemon.csv")
data.head(2)
data = data.set_index('#')
# also you can use --> data.index = data['#']
data.head(2)
print(data["HP"][1],"--> square brackets") # using square brackets
print(data.HP[1],"--> column attribute and row label") # using column attribute and row label
print(data.loc[1,["HP"]]) # using loc accessor (Dataframe)  ---{   print(data.loc[1,"HP"])--->Series   }---
print(data.loc[2:4,["Legendary","Type 1"]])
print("-"*25)
print(data.loc[2:4,"Total":"Defense"])
#data.loc[1:5,"Attack":"Sp. Def"]
data.loc[5:1:-1,"Attack":"Sp. Def"]
dataApply = pokedata.head()
def div(n):
    return n/2
#-------------------------------------#
print(dataApply.HP.apply(div))
# Like map()
print(list(map(div,dataApply.HP)))
#-------------------------------------#
dataApply.HP.apply(lambda x: x/2)
countrydata = countrydata.set_index(["country","year"])
countrydata.head(24)
dictionary = {"Map":["Howling Abyss","Howling Abyss","Summoner's Rift","Summoner's Rift"],"Region":["Zaun","Noxus","Zaun","Noxus"],"Champion":["Urgot","Sion","Viktor","Darius"],"WinRate":[51.37,52.12,51.95,50.28],"Patch":"8.21"}# add Patch
league = pd.DataFrame(dictionary)
league.head()
league.pivot(index="Map", columns="Region", values="Champion")
league1 = league.set_index(["Map","Region"])
league1
#level determines indexes
league1.unstack(level=0)
league1.unstack(level=1)
league1
league2 = league1.swaplevel(0,1)
league2
league
#league.pivot(index="Map", columns="Region", values="Champion")
pd.melt(league,id_vars="Map",value_vars=["Champion","WinRate"])
league
league.groupby("Map").mean() # only WinRate is integer
# there are other methods like sum, std,max or min
league.groupby("Region").WinRate.max() 
#we can choose multiple features
#league.groupby("Region")[["WinRate","another integer column"]].max() 
# as you can see Region is object
# However if we use groupby, we can convert it categorical data. 
# Because categorical data uses less memory, speed up operations like groupby
league["Region"] = league["Region"].astype("category")
league["Map"] = league["Map"].astype("category")
league.info()