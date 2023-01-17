# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





import seaborn as sns

sns.set(style="darkgrid")



from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/seaborn-datasets/repository/mwaskom-seaborn-data-23ee2ba/diamonds.csv')
df.head()
df.info()
sns.relplot(x="carat",y="price",data=df,height=7)
sns.relplot(x="carat", y="price",data=df,height=7, alpha=0.25, edgecolor=None)
# Add "clarity" variable as color

sns.relplot(

            x="carat",

            y="price",

            hue="clarity", # added to color axis

            data=df,

            height=7,

            palette="Set1", # change color palette 

            edgeColor=None)
sns.relplot(

            x="carat",

            y="price",

            hue="clarity",

            size="depth",   ###

            style="color",  ###

            data=df,

            palette="CMRmap_r",

            edgecolor=None,

            height=7)
df1=df.iloc[:250]



sns.relplot(x="carat",y="depth",data=df1,kind="line",ci=None)
fmri=pd.read_csv('../input/seaborn-datasets/repository/mwaskom-seaborn-data-23ee2ba/fmri.csv')
fmri.head()
fmri.info()
sns.relplot(x="timepoint",y="signal",kind="line",data=fmri,height=7)
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri, ci="sd", height=7)
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri, ci=None, height=7)
sns.relplot(

            x="timepoint",

            y="signal",

            size="event",

            style="region",

            markers=True,

            kind="line",

            data=fmri,

            hue="region",

            height=7

            )



plt.savefig("graph2.png")
sns.relplot(x="timepoint",

            y="signal",

            col="region", # show region in columns

            data=fmri,

            height=7)
sns.relplot(x="timepoint",

            y="signal",

            col="region", # show region in columns

            row="event",  # show event in rows

            kind="line",

            data=fmri,

            height=7)
sns.relplot(x="timepoint", y="signal", 

            col="subject", kind="line",

            data=fmri)
sns.relplot(x="timepoint", 

            y="signal", 

            col="subject", 

            col_wrap=4,

            kind="line",

            data=fmri)
data=pd.read_csv('../input/videogamesales/vgsales.csv')
data.head()
data.info()
data.dropna(how="any",inplace = True)

data.info()

data.Year = data.Year.astype(int)
# 



platform_count = Counter(data.Platform)

most_platform=platform_count.most_common(20)

platform_name,count = zip(*most_platform)

platform_name,count = list(platform_name),list(count)



# visualization



plt.figure(figsize=(15,10))

ax=sns.barplot( x = platform_name, y = count, palette = 'rocket')

plt.xlabel('Platform')

plt.ylabel('Frequency')

plt.title('Most common 20 of Platform')

plt.show()

# 2013-2016 

first_filter=data.Year>2012

second_filter=data.Year<2017

new_data=data[first_filter&second_filter]



plt.figure(figsize=(15,10))

sns.catplot(x="Year",y="Global_Sales",kind="bar",

            hue="Platform",

            data=new_data,

            edgecolor=None,

            height=8.27, aspect=11.7/8.27,ci=None)

plt.show()
#2010-2016

first_filter=data.Year>2009

second_filter=data.Year<2017

new_data1=data[first_filter&second_filter]





#visualization



sns.catplot(x="Year",y="NA_Sales",kind="point",

            data=new_data1,

            hue = "Platform",

            palette='Set1',

            ci = None,

            edgecolor=None,

            height=8.27, 

            aspect=11.7/8.27)

plt.show()
data1=data[['Year','Genre','Global_Sales']]

data1=data1.set_index('Year')

data2010=[]

data2010.append([sum(data1.loc[2010].Genre=='Shooter'),sum(data1.loc[2010].Genre=='Sports'), sum(data1.loc[2010].Genre=='Action'),sum(data1.loc[2010].Genre=='Role-Playing')])

data2010.append([sum(data1.loc[2011].Genre=='Shooter'),sum(data1.loc[2011].Genre=='Sports'), sum(data1.loc[2011].Genre=='Action'),sum(data1.loc[2011].Genre=='Role-Playing')])

data2010.append([sum(data1.loc[2012].Genre=='Shooter'),sum(data1.loc[2012].Genre=='Sports'), sum(data1.loc[2012].Genre=='Action'),sum(data1.loc[2012].Genre=='Role-Playing')])

data2010.append([sum(data1.loc[2013].Genre=='Shooter'),sum(data1.loc[2013].Genre=='Sports'), sum(data1.loc[2013].Genre=='Action'),sum(data1.loc[2013].Genre=='Role-Playing')])

data2010.append([sum(data1.loc[2014].Genre=='Shooter'),sum(data1.loc[2014].Genre=='Sports'), sum(data1.loc[2014].Genre=='Action'),sum(data1.loc[2014].Genre=='Role-Playing')])

data2010.append([sum(data1.loc[2015].Genre=='Shooter'),sum(data1.loc[2015].Genre=='Sports'), sum(data1.loc[2015].Genre=='Action'),sum(data1.loc[2015].Genre=='Role-Playing')])

data2010.append([sum(data1.loc[2016].Genre=='Shooter'),sum(data1.loc[2016].Genre=='Sports'), sum(data1.loc[2016].Genre=='Action'),sum(data1.loc[2016].Genre=='Role-Playing')])



df=pd.DataFrame(data2010,columns = ['Shooter' , 'Sports', 'Action','Role-Playing'])

df['Year']=[2010,2011,2012,2013,2014,2015,2016]



#visual



f,ax1 = plt.subplots(figsize =(20,10))



sns.pointplot(x='Year',y='Action',data=df,color='lime',alpha=0.7)

sns.pointplot(x='Year',y='Shooter',data=df,color='red',alpha=0.7)

sns.pointplot(x='Year',y='Sports',data=df,color='blue',alpha=0.7)

sns.pointplot(x='Year',y='Role-Playing',data=df,color='orange',alpha=0.7)





plt.xlabel('Years',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.text(5.7,240,'Action',color='lime',fontsize = 15,style = 'italic')

plt.text(5.7,230,'Shooter',color='red',fontsize = 15,style = 'italic')

plt.text(5.7,220,'Sports',color='blue',fontsize = 15,style = 'italic')

plt.text(5.7,210,'Role-Playing',color='orange',fontsize = 15,style = 'italic')

plt.grid()
iris=pd.read_csv('../input/iris/Iris.csv')
iris.head()
iris.info()
iris.corr()
sns.jointplot(x="SepalLengthCm",y="PetalLengthCm",data=iris)
sns.jointplot(x="SepalLengthCm",y="PetalLengthCm",data=iris, kind="hex", height=8)

sns.jointplot(x="SepalLengthCm",y="PetalLengthCm",data=iris, kind="kde",height=8)
iriscorr=iris.drop(["Id"],axis=1).corr()

iriscorr
#correlation map



f, ax = plt.subplots(figsize=(7,7))

sns.heatmap(iriscorr, annot=True, linewidths=0.2, fmt='.2f', ax=ax, cmap="rocket_r" )

plt.show()
nRowsRead = 1000 # specify 'None' if want to read whole file

cars = pd.read_csv('../input/cars-mini-dataset/cars.csv', delimiter=';', nrows = nRowsRead)

cars.dataframeName = 'cars.csv'

nRow, nCol = cars.shape

print(f'There are {nRow} rows and {nCol} columns')
cars.info()
cars.head()
f, ax = plt.subplots(figsize=(6,8))



sns.boxplot(x="Origin", 

            y="Horsepower", 

            data=cars,

            palette='Set2',

            ax=ax)
f, ax = plt.subplots(figsize=(8,10))

sns.boxplot(x="Origin", 

            y="Horsepower",

            hue="Cylinders",

            data=cars,

            palette='tab10',

            ax=ax)


sns.catplot(x="Origin", 

            y="Horsepower",

            hue="Cylinders",

            data=cars,

            palette='tab20',

            kind="box",

            height=8

            )
sns.catplot(x="Origin", 

            y="Horsepower",

            data=cars,

            palette='inferno',

            kind="boxen",

            height=8

            )

plt.savefig('graph.png')
fmri.head()
f, ax = plt.subplots(figsize=(10,8))

sns.stripplot(x="subject",

              y="signal",

              data=fmri,

              ax=ax,

              palette="hsv")
f, ax = plt.subplots(figsize=(10,8))

sns.stripplot(x="signal",

              y="subject",

              data=fmri,

              ax=ax,

              palette="hsv")
f, ax = plt.subplots(figsize=(10,8))

sns.stripplot(x="subject",

              y="signal",

              data=fmri,

              jitter=False,

              alpha=0.25,

              ax=ax,

              palette="hsv")

plt.savefig("graph1.png")
f, ax = plt.subplots(figsize=(8,8))

sns.swarmplot(x="region",

              y="signal",

              data=fmri

              )
f, ax = plt.subplots(figsize=(8,8))

sns.swarmplot(x="region",

              y="signal",

              hue="event",

              data=fmri,

              palette="rocket"

              )
f, ax = plt.subplots(figsize=(8,8))

sns.swarmplot(x="region",

              y="signal",

              data=fmri,

              palette="CMRmap",

              alpha=0.5

              )

sns.boxplot(x="region",

            y="signal",

            data=fmri,

            palette="Set1",

            )
df1.head()
f, ax = plt.subplots(figsize=(10,8))

sns.countplot(x="color",

              hue="cut",

              data=df1,

              edgecolor=None,

              palette="inferno",

              ax=ax)


sns.pairplot(iris, kind="reg",

             x_vars=["SepalLengthCm","PetalLengthCm","PetalWidthCm"],

             y_vars=["SepalLengthCm","PetalLengthCm","PetalWidthCm"],

             height=5)
sns.pairplot(iris, kind="reg",

             x_vars=["SepalLengthCm","PetalLengthCm","PetalWidthCm"],

             y_vars=["SepalWidthCm","PetalLengthCm","PetalWidthCm"],

             height=5)
f, ax = plt.subplots(figsize=(8,8))

sns.violinplot(x="Origin", y="Horsepower", data=cars, ax=ax, inner="points")
sns.catplot(x="Origin", y="Horsepower", data=cars, kind="violin", height=8)
cars["Old"] = cars.Year < 76
sns.catplot(x="Origin",

            y="Horsepower",

            hue = "Old",

            data=cars,

            split=True, # You need to turn on this parameter!

            inner = "stick", 

            kind="violin",

            palette="Blues",

            height=8

           )
sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", data=iris, height=7)
sns.regplot(x="PetalLengthCm", y="PetalWidthCm", data=iris,order=3)
sns.lmplot(x="SepalLengthCm",

           y="PetalWidthCm",

           hue="Species",

           data=iris,

           height=7)
cylinder8 = cars.loc[cars.Cylinders == 8]

cylinder5 = cars.loc[cars.Cylinders == 5]



f, ax = plt.subplots(figsize=(10,8))



sns.kdeplot(cylinder8.Horsepower, cylinder8.Weight, cmap="Reds", shade=True, shade_lowest=False, alpha=0.9)

sns.kdeplot(cylinder5.Horsepower, cylinder5.Weight, cmap="Blues", shade=True, shade_lowest=False, alpha=1)
f, ax = plt.subplots(figsize=(7,7))

sns.distplot(df1.depth, kde=False)
f, ax = plt.subplots(figsize=(7,7))

sns.distplot(df1.depth, hist=False, kde=True)
f, ax = plt.subplots(figsize=(7,7))

sns.distplot(df1.depth)