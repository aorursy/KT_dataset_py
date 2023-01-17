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
data=pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
data.head(10)
#Taking some information about the data of athletes
data.info()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Height.plot(kind = 'line', color = 'g',label = ' Height ',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')
data.Weight.plot(color = 'r',label = ' Weight ',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('ID of the athletes')              # label = name of label
plt.ylabel('cm')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = Height, y = Weight
data.plot(kind='scatter', x='Height', y='Weight',alpha = 0.5,color = 'red')
plt.xlabel('Height')              # label = name of label
plt.ylabel('Weight')
plt.title(' Height Weight Scatter Plot')            # title = title of plot
plt.show()
#Histogram graph of age of athletes
data.Age.plot(kind='hist',bins=50,figsize = (7,7),color = 'red')
plt.xlabel('Age')              
plt.ylabel('Amount')
plt.title(' The age of the athletes') 
plt.show()
#the filter1 shows the athletes who are older than 20.
filter1 = data['Age']>20    
data[filter1]

#the filter2 shows the athletes who are older than 20 and have gold medals.
filter2 =np.logical_and(data['Age']>20, data['Medal']=='Gold')
data[filter2]

data=pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
data.head()  # head shows first 5 rows
data.tail()  # tail shows first 5 rows
data.shape
data.info()
print(data['Name'].value_counts(dropna =True)) #dropna is for counting NaN values  
data.describe()#which gives the infos of features
data.boxplot(column='Weight',by = 'Age')
mydata=data.head()
mydata
melted_mydata=pd.melt(frame=mydata,id_vars=['ID'],value_vars=['City','Year','Sport','Name'])
melted_mydata
melted_mydata.pivot(index = 'ID', columns = 'variable',values='value')
# Firstly lets create 2 data frame
data1 = data.head()
data2= data.tail(3)
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True)#ignor index varsa alt alta indexler yoksa olduu gibi alır
conc_data_row
data1 = data['Sport'].head()
data2= data['Name'].head()

#data1

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 1 : adds dataframes in column
#conc_data_col
data.dtypes
#convertig data types
data['Sex'] = data['Sex'].astype('category')
data['Year'] = data['Year'].astype('float')
data.dtypes
data.info()
data["Name"].value_counts(dropna =False)#this function counts same values 
data["Height"].value_counts(dropna =False)
data1=data
#data1['Height'].dropna(inplace=True)
#assert 1==2 
#assert  data1['Height'].notnull().all()
data1["Height"].fillna('empty',inplace =  True)#changing non values to another ,inplace demek non valueleri ototmatik doldurur demektir
assert  data1['Height'].notnull().all()#checks critical logics
data1
country = ["Spain","France","Italy"]
population = ["11","12","13"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
#add new coulmns
df["capital"]=["madrid","paris","rome"]
df
df["coronatime"]=[250,200,300]
df
df.info()
df['population']=df['population'].astype(int)
df.info()

data=pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
data1=data.loc[1:1000,["Age","Height","Weight"]]
data1.plot()
data1.plot(subplots = True)
plt.show()
data1.plot(kind = "scatter",x="Height",y = "Weight")
plt.show()
data1.plot(kind="hist",y = "Weight",bins=100,range=(50,150))
plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Height",bins = 100,range= (120,220),ax = axes[0])
data1.plot(kind = "hist",y = "Weight",bins = 100,range= (50,150),ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))

import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of athlete data and add it a time list
data2 = data.head(6)
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16","1995-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")
data2 
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-5":"1993-03-16"])
data2.resample("A").mean()#"M" = month or "A" = year
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
#note that no sample on 1994 but it takes the mean of the prev and next year.
data2.resample("A").mean().interpolate("linear")
data=pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')

data.head()

data= data.set_index("ID")
data.head()
data["Team"][1]
# using column attribute and row label
data.Team[1]
# using loc accessor
data.loc[1,["Team"]]

data[["Name","Team"]]
# Difference between selecting columns: series and dataframes
print(type(data["Height"]))     # series
print(type(data[["Height"]]))   # data frames
# Slicing and indexing series
data.loc[1:10,"Name":"City"]   # 10 and "City" are inclusive
data.loc[10:1:-1,"Name":"City"]   # 10 and "City" are inclusive
boolean = data.Height > 160
data[boolean]
# Combining filters
first_filter = data.Height > 160
second_filter = data.Weight > 80
data[first_filter & second_filter]
# Filtering column based others
data.Height[data.Weight<80]
def div(n):
    return n/2
data.Height.apply(div)
# Or we can use lambda function
data.Height.apply(lambda n : n/2)
# Defining column using other columns
data["strenght"] = data.Height*data.Weight/100
data.head()
# our index name is this:
print(data.index.name)
# lets change it
data.index.name = "index_name"
data.head()
# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to data3 then change index 
data3 = data.head(800)
# lets make index start from 100. It is not remarkable change but it is just example
data3.index = range(100,900,1)
data3.index.name = "index_name"
data3.head(10)

data = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
data.head()
data1=data.set_index(["Name","ID"])
data1.head(50)
dic = {"treatment":["A","A","B","B"],"disease":["Corona","Flue","Corona","Flue"],"# of death":[10,45,5,9],"# of recovery":[15,4,72,65]}
df = pd.DataFrame(dic)
df
# pivoting
df.pivot(index="treatment",columns = "disease",values="# of death")
df1 = df.set_index(["treatment","disease"])
df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)
df2
df
pd.melt(df,id_vars="treatment",value_vars=["# of death","# of recovery"])
df

# according to treatment take means of other features
df.groupby("treatment").mean()   # mean is aggregation / reduction method
# there are other methods like sum, std,max or min
df.rename(columns={'# of death':'#_of_death','# of recovery':'#_of_recovery'}, 
                 inplace=True)
df.groupby("treatment")['#_of_death'].max()#or 
# df.groupby("treatment").#_of_death.max() not work because of # sign which is for comments


df_4=df.groupby("treatment")[["#_of_death","#_of_recovery"]].min() 
df_4
df.info()