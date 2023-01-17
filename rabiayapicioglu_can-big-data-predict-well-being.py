from IPython.display import Image

import os

!ls ../input/



Image("../input/imageshappiness1/BigData.png")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data2015=pd.read_csv('../input/world-happiness/2015.csv')

#data2016=pd.read_csv('../input/2016.csv')

#data2017=pd.read_csv('../input/2017.csv')
print( data2015.info() )



#print( data2016.info() )

#since they all have same info()

#print( data2017.info() )



data2015.head()
print(" 2015 Correlation of data ")

data2015.corr() 



#print(" 2016 Correlation of data ")

#print( data2016.corr() )



#print(" 2017 Correlation of data ")

#print( data2017.corr() )
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data2015.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
from IPython.display import Image

import os

!ls ../input/



Image("../input/imageshappiness1/happy.png")
data2015.head()
from IPython.display import Image

import os

!ls ../input/



Image("../input/imageshappiness1/Happiness.png")
data2015.tail()
for each in data2015.columns:

    print( each )


#try to set index to dataframe

fig, axes = plt.subplots(figsize=(10, 10),nrows=2, ncols=2)



data_updated=data2015.rename( index=str ,columns={"Happiness Rank":"Happiness_Rank","Standard Error":"Standard_Error"})

data_2015U=data_updated.rename( index=str ,columns={"Happiness Score":"Happiness_Score"})

data_2015U=data_2015U.rename( index=str,columns={"Economy (GDP per Capita)":"Economy","Dystopia Residual":"Dystopia_Residual","Health (Life Expectancy)":"Health","Trust (Government Corruption)":"Trust"})



data_2015U.sort_values(by=['Happiness_Score'])

#print(data_2015U.loc[:,['Country','Happiness_Score']])

plt.legend(loc='upper right') 



data_2015U=data_2015U.set_index('Happiness_Score')

data_2015U.Standard_Error.plot(ax=axes[0,0],kind = 'line', color = 'red',title = 'Happiness Score',linewidth=1,grid = True,linestyle = ':')

data_2015U.Family.plot( ax=axes[0,1],kind='line' ,color='green' ,title='Family' ,linewidth=1 , grid=True ,linestyle=':' )

data_2015U.Economy.plot( ax=axes[1,0],kind='line' ,color='yellow', title='Economy',linewidth=1,grid=True ,linestyle=':' )

data_2015U.Health.plot( ax=axes[1,1],kind='line' ,color='blue', title='Health',linewidth=1,grid=True ,linestyle=':' )



    # legend = puts label into plot

              # label = name of label



          # title = title of plot

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



data2015=pd.read_csv('../input/world-happiness/2015.csv')



#fig, axes = plt.subplots(figsize=(10, 10),nrows=2, ncols=2)



data_updated=data2015.rename( index=str ,columns={"Happiness Rank":"Happiness_Rank"})

data_2015U=data_updated.rename( index=str ,columns={"Happiness Score":"Happiness_Score"})

data_2015U=data_2015U.rename( index=str,columns={"Economy (GDP per Capita)":"Economy","Health (Life Expectancy)":"Health","Trust (Government Corruption)":"Trust"})





f,ax = plt.subplots(figsize=(30, 30))



Western_Europe=data_2015U[ data_2015U.Region=='Western Europe']

North_America=data_2015U[ data_2015U.Region=='North America']

Australian_New_Zealand=data_2015U[ data_2015U.Region=='Australia and New Zealand']

Middle_East_and_Northern_Africa=data_2015U[ data_2015U.Region=='Middle East and Northern Africa']

Latin_America_and_Caribbean=data_2015U[ data_2015U.Region=='Latin America and Caribbean']

Southeastern_Asia=data_2015U[ data_2015U.Region=='Southeastern Asia']

Central_and_Eastern_Europe=data_2015U[ data_2015U.Region=='Central and Eastern Europe']

Eastern_Asia=data_2015U[ data_2015U.Region=='Eastern_Asia']

#Sub_Saharan_Africa=data_2015U[ data_2015U.Region=='Sub Saharan Africa']

Southern_Asia=data_2015U[ data_2015U.Region=='Southern Asia']





for each in range(0,len(Central_and_Eastern_Europe.Country)):

    x = Central_and_Eastern_Europe.Happiness_Score[each]

    y = Central_and_Eastern_Europe.Freedom[each]    

    plt.scatter( Central_and_Eastern_Europe.Happiness_Score,Central_and_Eastern_Europe.Freedom,color='magenta',linewidth=1)

    plt.text(x, y,Central_and_Eastern_Europe.Country[each], fontsize=15)





for each in range(0,len(Southern_Asia.Country)):

    x = Southern_Asia.Happiness_Score[each]

    y = Southern_Asia.Freedom[each]    

    plt.scatter( Southern_Asia.Happiness_Score,Southern_Asia.Freedom,color='yellow',linewidth=1)

    plt.text(x, y,Southern_Asia.Country[each], fontsize=15)

    

for each in range(0,len(Western_Europe.Country)):

    x = Western_Europe.Happiness_Score[each]

    y = Western_Europe.Freedom[each]    

    plt.scatter( Western_Europe.Happiness_Score,Western_Europe.Freedom,color='red',linewidth=1)

    plt.text(x, y, Western_Europe.Country[each], fontsize=15)

    

for each in range(0,len(North_America.Country)):

    x = North_America.Happiness_Score[each]

    y = North_America.Freedom[each]    

    plt.scatter( North_America.Happiness_Score,North_America.Freedom,color='blue',linewidth=1)

    plt.text(x, y, North_America.Country[each], fontsize=15)



    

for each in range(0,len( Middle_East_and_Northern_Africa.Country)):

    x =Middle_East_and_Northern_Africa.Happiness_Score[each]

    y =Middle_East_and_Northern_Africa.Freedom[each]    

    plt.scatter(  Middle_East_and_Northern_Africa.Happiness_Score, Middle_East_and_Northern_Africa.Freedom,color='purple',linewidth=1)

    plt.text(x, y,  Middle_East_and_Northern_Africa.Country[each], fontsize=15)



plt.title("Happiness Score-Freedom Scatter Plot")

plt.xlabel("Happiness Score",fontsize=20)

plt.ylabel("Freedom",fontsize=20)











melted = pd.melt(frame=data_2015U,id_vars = 'Country', value_vars= ['Generosity','Dystopia_Residual'])

melted.loc[:10]
data_2015U1=data_2015U.head()

data_2015U2=data_2015U.tail()



concat_data_row=pd.concat([data_2015U1,data_2015U2],axis=0,ignore_index=True)



concat_data_row
data1 = data_2015U.loc[:,["Health","Trust","Freedom"]]

data1.plot()
data1.plot(subplots = True)

plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2)



data_2015U.plot(ax=axes[0,0],kind = "scatter",x="Happiness_Score",y="Freedom",color="blue")

data_2015U.plot(ax=axes[0,1],kind = "scatter",x="Happiness_Score",y="Family",color="red")

data_2015U.plot(ax=axes[1,0],kind = "scatter",x="Happiness_Score",y="Economy",color="yellow")

data_2015U.plot(ax=axes[1,1],kind = "scatter",x="Happiness_Score",y="Generosity",color="pink")
fig, axes = plt.subplots(nrows=2,ncols=1)

data_2015U.plot(kind = "hist",y = "Happiness_Score",bins = 50,range= (0,250),normed = True,ax = axes[0])

data_2015U.plot(kind = "hist",y = "Happiness_Score",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

#plt.savefig('graph.png')

plt