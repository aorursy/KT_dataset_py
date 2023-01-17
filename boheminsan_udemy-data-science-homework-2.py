# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
wdata = pd.read_csv('../input/countries of the world.csv')
wdata.info()
wdata.head(20)
wdata.columns
def enbuyuk(count=5):
    """ return a list of largest 5 countries"""
    return wdata.sort_values(by=['Area (sq. mi.)'],ascending=False).head(count)

enbuyuk()
def coast(country):
    #columns = [c.replace(' ', '_') for c in wdata.columns]
    #columns = [c.replace('.', '_') for c in wdata.columns]
    Country=wdata.at[country-1,'Country']
    area=wdata.at[country-1,'Area (sq. mi.)']
    def multiply():
        sqkm=round(area*0.38610)
        print(Country)
        print("has", area, "km sq area")
        print("has", sqkm, "mi sq area")
    multiply()
coast(12)
def ulkeler(*args):
    for i in args:
        print(i)
        
countries=tuple(wdata.iloc[:, wdata.columns.get_loc('Country')])
#ulkeler(countries)

dict=wdata.set_index('Country').to_dict()['Population']
def f(**kwargs):
    for key, value in kwargs.items():
        print(key, " ", value)
    
f(**dict)
pop=list(wdata.iloc[:, wdata.columns.get_loc('Population')])
pop2=[]

for i in range(len(pop)):
    pop2.append(int(pop[i]))
ort=sum(pop2)/len(pop2)
#print(ort)
kalabalik = list(filter(lambda x: (x>ort) , pop2))
print(kalabalik)
# zip example
list1 = [1,2,3,4,5,6,7,8,9,10]
list2 = ['A','D','C','D','C','B','B','D','C','A']
z = zip(list1,list2)
z_list = list(z)
print("yanıtlar", z_list)  
pop=list(wdata.iloc[:, wdata.columns.get_loc('Population')])
pop2=[]

for i in range(len(pop)):
    pop2.append(int(pop[i]))
    
ort=sum(pop2)/len(pop2)
wdata["Nufus"] = ["kalabalik" if i > 3*ort else "dusuk" for i in pop2]
wdata.reindex(columns=["Country","Nufus","Population"]).head(10)
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(wdata.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
