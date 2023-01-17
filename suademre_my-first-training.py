# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns   # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2016.csv")  #first we are input the data 
data.shape   #we can check how many columns and rows are there 
data.columns     #name of the columns in dataset
#There is a simple problem in dataset. when there is a space between columns of name, we muss put '_' or write together.
data.columns = [each.lower() for each in data.columns] 
data.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.columns]
data.columns    #our new database
data.rename(columns={'economy_(gdp':'economy_gdp','health_(life':'health_life','trust_(government':'trust_government'},inplace=True) 
data.columns
data.info()     # we learn information about database
data.isnull().sum()       # we are checking for missing data, we haven't got missing data
data.describe()    #we are describe the data
data.head(10)   # first 10 data in database
print(len(data.region.unique())) 
data.region.unique()   #check about how many different regions
data_vc = data['region'].value_counts()      #We check which region is the most in dataset.
data_vc.head(10)     #first 10 data show 
data.corr()     # we are looking at the correlation between numerical values
f,ax = plt.subplots(figsize=(18,18))    #correlation between numerical values' maps
sns.heatmap(data.corr() , annot = True, linewidths = .5, fmt= '.1f', ax=ax)
plt.legend()
plt.show() 
data.plot(kind = 'scatter', x='happiness_score', y='upper_confidence',alpha=0.5,color='r')
plt.xlabel("happiness_score")
plt.ylabel("upper_confidence")
plt.title("happiness - confidence Scatter Plot")
plt.show() 
west_eu=data[data.region=="Western Europe"]  
west_eu.head(10)  
cen_east_eu=data[data.region=="Central and Eastern Europe"] #i wanna work with Central and Eastern Europe, so i create another columns for Central and Eastern Europe
cen_east_eu.head(10) 
ort = sum(data['happiness_score']) / len(data['happiness_score'])
print("average_life_standard:",ort)
data['happiness_score_level'] = ["easy life" if i > ort else "hard life" for i in data['happiness_score']]
data.loc[:100] 
data['yasam'] = ["high life" if i<15 else "middle life" if (30>i) else "low life" for i in data ['happiness_rank']]
data.head(40) 
data[(data['happiness_rank']<15) & (data['economy_gdp']>1.35)] 
#data[np,logical_and(data['happiness_score_level']= 'yasanabilir' , data['yasam']='orta yasam')] 
data['health_life'].corr(data['economy_gdp'])  #Correlation between health life and economy 
data.generosity.plot(kind = 'hist',bins = 30,figsize = (10,10))  #we learn generosity frequency
plt.show 
data.region = [each.lower() for each in data.region] 
data.region = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.region]
data.region.unique() 
data.plot(kind = 'scatter', x='economy_gdp',y='generosity',alpha=0.8,color='green')
plt.xlabel("Economy")
plt.ylabel("Generosity")
plt.title('Economy-Generosity')
plt.show()
data.plot(kind = 'scatter', x='economy_gdp',y='happiness_score',alpha=0.8,color='green')
plt.xlabel("Economy")
plt.ylabel("Generosity")
plt.title('Economy-Happiness_score')
plt.show() 
#Dataframe from Dictionary
country = ["Italy","Germany"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df 
df["capital"]=["Roma","Berlin"]
df 
df["income"]=0     #Broadcasting entire column
df 
data1=data.loc[:,["health_life","economy_gdp","freedom"]]    #Plot
data1.plot() 
#Subplots
data1.plot(subplots = True)
plt.show() 
#scatter plot   we are loking for the corralation between two columns
data1.plot(kind = "scatter", x="health_life",y="economy_gdp")
plt.show() 
#hist 
data.plot(kind = "hist", y="happiness_score",bins=30,range = (0,10),normed = True)
plt.show()   
#histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data.plot(kind = "hist",y="happiness_score",bins = 30,range=(1,10),normed = True,ax = axes[0])
data.plot(kind = "hist",y="happiness_score",bins = 30,range=(1,10),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show() 
#we are creating the date which is the string,however we want change datetime object
time_list = ["2018-03-08","2018-04-12"]
print(type(time_list[1]))  #now date is string
datatime_object = pd.to_datetime(time_list)
print(type(datatime_object))  #now date is datetime object  
data2=data.head() 
date_list = ["2017-01-10","2017-02-10","2017-03-10","2018-03-15","2018-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
#lets make date as index
data2 = data2.set_index("date")
data2   
#Now we can select according to our date index
print(data2.loc["2018-03-16"])
print(data2.loc["2017-03-10":"2018-03-16"])  
#Now we can select according to our data's columns
print(data2.loc[:,"happiness_score"]) 
print(data2.loc[:, "happiness_score":"freedom"]) 
print(data2.loc[:,:"happiness_score"]) 
print(data2.loc[::-1,:]) 
data2.resample("Y").mean()   #we are learning how many years are there in data
#lets resample with month
data2.resample("M").mean() 
# there are alot of nan because data2 does not include all months. We can solve this problem next step
#we can solve this problem with interpolate in happiness_rank
data2.resample("M").first().interpolate("linear") 
#second way is interpolatewith mean
data2.resample("M").mean().interpolate("linear") 
data2["happiness_rank"][2] 
data2.happiness_rank[2] 
data2[["happiness_rank","happiness_score"]] 
data.loc[1:10,"happiness_score"]  #10 and happiness_score are inclusive 
#slicing data frame
data.loc[10:1:-1,"happiness_rank":"economy_gdp"] 
#From something to end
data.loc[1:10,"economy_gdp":] 
boolean=data.happiness_score > 7.0  #Creating boolean series
data[boolean] 
data.head(50)  
# Combining filters 
first_filter = data.happiness_score > 7
second_filter = data.economy_gdp >1.47
data[first_filter & second_filter] 
# Filtering column based others
data.happiness_score[data.economy_gdp>1.47] 
# Plain Pyton function 
def div(n):
    return n/2
data.happiness_score.apply(div) 
# if you want we can use second way as lambda function
data.happiness_score.apply(lambda n:n/2) 
#Defining column using other columns
data["confidence_interval"] = data.upper_confidence - data.lower_confidence 
data.head() 
# or
data["confidence_interval_avrg"] = (data.upper_confidence + data.lower_confidence) / 2
data.head() 
#we are looking our index name
print(data.index.name) 
#we are changing index name
data.index.name = "index_name"
data.head() 
#If we want to modify index we need to change all of them
data.head() 
data3=data.copy()  # i wanna copy because i dont want to change my orginal data
data3.index = range(100,257,1) 
data3.head()  
data1 = data.set_index(["region","country"])
data1.head (50)    
dic={"treatment" : ["A","A","B","B"], "gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df=pd.DataFrame(dic)
df 
#Pivoting 
df.pivot(index="treatment",columns = "gender",values="response") 
#unstack
df1 = df.set_index(["treatment","gender"])
df1 
#level determines indexes
df1.unstack(level=0) 
df1.unstack(level=1) 
#Change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2 
# Reverse of pivoting 
df
#df.pivot(index="treatment", columns="gender",values="response")
pd.melt(df,id_vars="treatment",value_vars=["age","response"]) 
df
df.groupby("treatment").mean() 
df.groupby("gender").mean() 
df.groupby("age").max() 
df.groupby("treatment") [["age","response"]].min() 


