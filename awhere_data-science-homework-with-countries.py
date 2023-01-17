# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualization
import seaborn as sns #for visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/countries of the world.csv")  #reading csv with pandas
data.info() #column's name,type,count and row's count
#shape gives number of rows and columns
data.shape
data.head() #it shows first 5 row or you can show 10 rows head(10) etc..
# some columns's names have gap, brackets, dot etc.. we have to change them for use them in some methods.
data.columns
data.rename(columns={"Area (sq. mi.)": "Area", "Pop. Density (per sq. mi.)":"Pop_Density",
                        "Coastline (coast/area ratio)":"Coastline","Net migration":"Net_migration",
                        "Infant mortality (per 1000 births)":"Infant_mortality","GDP ($ per capita)":"GPD",
                        "Literacy (%)":"Literacy","Phones (per 1000)":"Phone_using","Arable (%)":"Arable",
                        "Crops (%)":"Crops","Other (%)":"Other"},inplace = True)
#We are using inplace=True to change column names in place.
data.columns
# as we can see many columns' entry is  number but they include ","(comma) so pandas describe them object
# so we replace ","(comma) to "."(dot) then change their types float with "astype"
data.dtypes 
data.Literacy = data.Literacy.str.replace(",",".").astype(float)
data.Pop_Density = data.Pop_Density.str.replace(",",".").astype(float)
data.Coastline = data.Coastline.str.replace(",",".").astype(float)
data.Net_migration = data.Net_migration.str.replace(",",".").astype(float)
data.Infant_mortality = data.Infant_mortality.str.replace(",",".").astype(float)
data.Phone_using = data.Phone_using.str.replace(",",".").astype(float)
data.Arable = data.Arable.str.replace(",",".").astype(float)
data.Crops = data.Crops.str.replace(",",".").astype(float)
data.Birthrate = data.Birthrate.str.replace(",",".").astype(float)
data.Deathrate = data.Deathrate.str.replace(",",".").astype(float)
data.Agriculture = data.Agriculture.str.replace(",",".").astype(float)
data.Industry = data.Industry.str.replace(",",".").astype(float)
data.Service = data.Service.str.replace(",",".").astype(float)
data.Other = data.Other.str.replace(",",".").astype(float)
data.Climate = data.Climate.str.replace(",",".").astype(float)
#checking 
data.dtypes
data.describe() #only numeric feature (float,int)
#boxplot = visualize basic statistics (outliers,max,min)

#first black line at top  is max
#first blue line at top is  Q3 
#green line is median 
#second blue line at bottom is Q1
#second black line at bottom is min
#circle at bottom is lower fence
#upper fence

data.boxplot(column = "Literacy")
d1 = data.head(10)

#d1.loc[:,"Area":"GPD"] #first select is row ":" for all row , second is column
#d1.loc[:,["Area","GPD"]]
#d1.loc[:2,:]
#d1.loc[::-1,"Area"]
#d1.loc[::-1,["Area"]]
#d1.iloc[:,:2] #when we use iloc (integer-location) ıf we select 1. column we have to write 2
#d1.iloc[1:4,2:5]

d1 = data.head(3)
d2 = data.tail(3)
# axis = 0 vertical(working with row) concat  axis=1 horizontal(working with column) 
#ignore_index = True new indexing numbers
data_concat = pd.concat([d1,d2], axis=0, ignore_index = True)
data_concat 
d1 = data.loc[2:5,"Country":"Area"]
d2 = data.loc[2:5,"GPD"]
#axis = 1 working with column you can see it when we use ignore_index = True or False
data_concat = pd.concat([d1,d2],axis=1 ,ignore_index = False) 
data_concat
#frequency of climates 
#dropna = False it shows nan values frequency
data.Climate.value_counts(dropna = False)
#drop nan values
data.Climate = data.Climate.dropna() 

#checking with assert we can check alot thing in data
assert data.Climate.notnull().all()  #return nothing because it is true 
#if it is false return error
#assert 1==2 #example
#change nan values. 
#if you drop nan values then you use fillna you change all values
data.Climate = data["Climate"].fillna(value=0)  
assert data.Climate.notnull().all()
d1 = data.head(10)
d1
#value_vars = what we want to melt
#id_vars = what we dont want to melt
melt_d1 = pd.melt(frame = d1, id_vars = "Country", value_vars = ["Population","GPD","Literacy"])
melt_d1
#reverse of melting
melt_d1.pivot(index = "Country", columns = "variable", values = "value")
data.head(7)
mean_area = data.Area.mean()
data["area_level"] = ["big" if i>mean_area else "small" for i in data.Area]
data.loc[:10,["area_level","Area"]]
data.head()
#Line plot  is better when x axis is time.
#alpha = opacity,  linewidth=width of line, figsize = size of the plot(x,y)
data.Birthrate.plot(kind = "line",x= "Birthrate",  color = "g", label = "Birthrate", linewidth = 3, alpha = 0.6, grid = True,
                   linestyle = "-", figsize = (20,5))
data.Deathrate.plot(kind = "line",x= "Deathrate",  color = "black", label = "Deathrate", linewidth = 3, alpha = 0.6, grid = True,
                   linestyle = "-", figsize = (20,5))

plt.legend(loc ="upper right") #it shows label into plot
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Line plot")
plt.show()
#Scatter plot is better when there is correlation between two variables

data.plot(kind = "scatter", x = "GPD", y = "Literacy", alpha = 0.7, color = "r", grid = True, figsize = (8,8))
plt.xlabel("GPD")
plt.ylabel("Literacy")
plt.title("scatter plot")
plt.show()
#histogram is better when we need to see distribution of numerical data.
#bins = number of bar in figure
data.Phone_using.plot(kind = "hist", bins = 50, figsize = (20,5),  label="Phone_using")
plt.legend()
plt.title("Histogram Plot")
plt.show()


#clf = cleans it up again you can start a fresh
data.Phone_using.plot(kind = "hist", bins = 60, figsize = (10,10), grid = True)
plt.clf()
#we cant see plot due to clf
data.corr()
#correlation map
#annot = True means we can see numbers in square, fmt = number of digits after comma
f,ax = plt.subplots(figsize = (20,10))
sns.heatmap(data.corr(),annot = True, linewidth = 0.5, fmt = ".1f", ax=ax)  #seaborn 
plt.show()



