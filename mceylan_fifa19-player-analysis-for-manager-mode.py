'''

I am a newbie and this is my practice area.I will uptade this analysis as I learn.

I am doing this analysis with the purpose of helping me and other fifa manager mode lovers.

'''







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
import matplotlib.pyplot as plt

import seaborn as sns 
data = pd.read_csv("../input/fifa19/data.csv") # importing data
data.info() #general ingfo
data.corr()





## gives us the correlations between features
#correlation map



f,ax = plt.subplots(figsize=(30,30))

sns.heatmap(data.corr(), annot=True , linewidth = 1 , fmt ='.1f', ax = ax) 

plt.show() 







#First look of Data



data.head(10)
#features



data.columns



## Lots of features has spaces and upper-lower letters together.
#Data with Numerical Fetures - ıt may help while operating.

# Deleting all str Features except "Name"



k =0

numericaldata = data   

while   k<89  :

    if (data.columns[k] == "Name"):  ## bütün str olan featureları yoket

        k = k+1

    elif(type(data.iloc[0,k]) == str):

        numericaldata = numericaldata.drop([data.columns[k]],axis =1)

        k=k+1

    else :

        k=k+1

del k





# numericaldata.columns 
#%% Upper-Lower and spaces Problem



data.columns = [each.split()[0]+each.split()[1] and each.lower() if len(each.split())>1 else each.lower() for each in data.columns]

numericaldata.columns = [each.split()[0]+each.split()[1] and each.lower() if len(each.split())>1 else each.lower() for each in numericaldata.columns]



# Doing this we get rid of spaces and upper letter in data's features.

# Now our data is better to operate.
#lineplot



data.potential.plot(kind = 'line' , color = 'g' ,linewidth = 1 ,label = "Potential", alpha = 0.5 ,grid = True , linestyle = ':',figsize = (20,20))

data.overall.plot( color = 'b' ,linewidth = 1 ,label = "Overall", alpha = 1 ,grid = True )

plt.title("Potential & Overall LinePlot")

plt.xlabel("index")

plt.legend(loc ='lower left')

plt.show()
# ScatterPlot



# let s take a look into the wonderkids candidates.



filter1 = data.age < 21 

filteredage = data[filter1] #players which are below 21



filteredage.plot(kind = "scatter",x = "overall",y = "potential", alpha = 0.7,color ="b",figsize = (20,20), s = 75 )

plt.xlabel("Overall")

plt.ylabel("Potential")

plt.title("Overall-Potential")

plt.show()



#Wonderkids are located upper right side on the grapchic.



#Also the probable best player of the future is obviously seen in the graph.

#He has around 90s overall and potential.
# To detect specifically wonderkids , let's add some other filters



filter2 = data.potential > 85

filter3 = data.overall > 70



wonderkidslist = data[filter1 & filter2 & filter3] 



#print(wonderkidslist)



# As you expect , K. Mbappé is shining top of the list.

#For best of best;



filter4 = data.potential >= 90

bestofwonderkids = wonderkidslist[filter4]



# bestofwonderkids.name    #  --> That is the list for manager mode players :)  





# "Vinícius Júnior" has great age and potential , tranferings this player about 20-30M 

#  will return your team as more than 100M in several years.
#Distributon of Age



data.age.plot(kind= "hist", color = "r", bins = 60, figsize = (20,20) )

plt.show()
#Distribution of Overall



data.overall.plot(kind = "hist", bins = 100, color = "black",figsize = (20,20) )

plt.show()
#hist2d overall-potential



plt.hist2d(data.potential, data.overall,bins = 25)

plt.colorbar()

plt.xlabel("Potential")

plt.ylabel("Overall")

plt.show()