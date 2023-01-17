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
print( check_output(["ls","../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data15 = pd.read_csv("../input/2015.csv") #we needs to data read.
data16 = pd.read_csv("../input/2016.csv")
data17 = pd.read_csv("../input/2017.csv")
data15.info()


data15.corr()

f,ax = plt.subplots( figsize = (15,15))
sns.heatmap( data15.corr() , annot = True , linewidth = 1 , fmt =(".1f"), ax=ax)
plt.show()

data15.columns

data15.head(15)
#we need to correct inappropriate column writings for us
data15.columns = [  each.split()[0]+"_"+each.split()[1] if (len(each.split()) > 1) else each for each in data15.columns]
data16.columns = [ each.split()[0]+"_"+each.split()[1] if (len(each.split()) > 1 ) else each for each in data16.columns]
data17.columns = [ each.split()[0]+"_"+each.split()[1] if (len(each.split()) > 1) else each for each in data17.columns]
data15.columns
#line plot
data15.Generosity.plot( kind = "line", color = "green", label="GPD 15",linewidth=1, grid= True, alpha =1, linestyle=":",figsize = (15,15))
data16.Generosity.plot( kind = "line", color = "red", label="GPD 16",linewidth=1, grid= True, alpha =0.8, linestyle="-.",figsize = (15,15))
data17.Generosity.plot( kind = "line", color = "orange", label="GPD 17",linewidth=1, grid= True, alpha =0.5, linestyle="-",figsize = (15,15))
plt.legend()
plt.title("Generosity change a year")
plt.xlabel("Country")
plt.ylabel("Values")
plt.show()


#scatter plot
#x = Happiness_Score , y = Happiness_Rank
data15.plot( kind = "scatter", x = "Happiness_Score" , y = "Happiness_Rank" , color = "orange", grid = True , figsize = (15,10), alpha = 0.5)

plt.xlabel("Happiness_Score")
plt.ylabel("Happiness_Rank")
plt.title("proportionality(orantililik)")
plt.show()
# histogram
data15.Happiness_Score.plot( kind ="hist", color = "green", alpha = 0.5, grid = True, bins = 20 , figsize = (15,10) )
plt.title("Happiness_Score histogram")
plt.xlabel("Happiness_Score")
plt.show()
data15.Happiness_Score.plot( kind ="hist", bins = 20, grid = True, color = "red")
plt.clf() #hides plots.
#create dictionary and look its keys and values
dictionary = { "Germany" : "100", "Turkey" : "80", "USA" : "180", "Nedherlands" : "70"}
print(dictionary.keys())
print(dictionary.values())
dictionary["France"] = "95"
print(dictionary)
del dictionary["Nedherlands"]
print(dictionary)
dictionary.clear()
print(dictionary)

del dictionary
#dictionary
data15 = pd.read_csv("../input/2015.csv")
data15.columns = [ each.split()[0]+"_"+each.split()[1] if (len(each.split()) > 1) else each for each in data15.columns]
data15.head(15)
series = data15["Standard_Error"]
print(type(series))
data_frame = data15[["Standard_Error"]]
print(type(data_frame))
print(10.5<10)
print(11!=11.1)
print(11==(5+6))

print(True and True)
print(True and False)
print(False and False)

print(True or True)
print(True or False)
print(False or False)
x =  data15["Economy_(GDP"] > 1.35 
data15[x]
data15[ np.logical_and(data15["Economy_(GDP"] > 1.35 , data15["Happiness_Score"] > 7.1 )]

data15[(data15["Economy_(GDP"] > 1.35) & (data15["Happiness_Score"] > 7.1)]
i = 0
while (i !=10) :
    print("i : ",i)
    i += 2
print("allowable i :",i)

liste = [1,2,3,4,5,6]
for i in liste :
    print("i :",i)
print()

for index , value in enumerate(liste):
    print(index,":",value)
print("")

for index , value in data15[["Economy_(GDP"]][0:2].iterrows():
    print(index,":",value)
print("")

dictionary = { "Turkey" : "İstanbul", "USA" : "Washington"}
for key , value in dictionary.items():
    print(key,":",value)
print("")