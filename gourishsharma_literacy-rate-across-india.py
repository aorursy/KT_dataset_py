import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt    

import seaborn as sns

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df=pd.read_csv('/kaggle/input/govt-of-india-literacy-rate/GOI.csv')  #read the data 
df.head()
df.isnull().sum()    #check null values  
df.info()          #check datatypes 
df.describe()
#column names are too long so i alter the names 

df.rename(columns={"Literacy Rate (Persons) - Total - 2001":"Total - 2001","Literacy Rate (Persons) - Total - 2011":"Total - 2011","Literacy Rate (Persons) - Rural - 2001":"Rural - 2001","Literacy Rate (Persons) - Rural - 2011":"Rural - 2011","Literacy Rate (Persons) - Urban - 2001":"Urban - 2001","Literacy Rate (Persons) - Urban - 2011":

                   "Urban - 2011"},inplace=True)
df.head()
df.iloc[:1]     #to check overall literacy rate in India i need only first row

#plot the graph to check overall literacy rate across India 

df.iloc[:1].plot(kind='bar',title='Overall Literacy Rate In India',color=['grey','blue'],figsize=(10,8))

plt.xlabel('Total        Rural        Urban ')

#plt.grid()

plt.legend(["Total-2001","Total-2011"],loc='upper left')
#percentage change in litercy rate over the decade.

df['Total - Per. Change'] = (df.loc[:,'Total - 2011'] - 

                df.loc[:,'Total - 2001'])/df.loc[:,'Total - 2001']

df['Rural - Per. Change'] = (df.loc[:,'Rural - 2011'] - 

                df.loc[:,'Rural - 2001'])/df.loc[:,'Total - 2001']

df['Urban - Per. Change'] = (df.loc[:,'Urban - 2011'] - 

                df.loc[:,'Urban - 2001'])/df.loc[:,'Total - 2001']
df.head()
df.drop([0],axis='rows',inplace=True)     #Now i dont need the overall data of Country 
df.rename(columns={"Country/ States/ Union Territories Name":"States/ Union Territories"}

          ,inplace=True)                  #remove country name
df.head()
df.sort_values('Total - 2001',inplace=True) 
df.head()
x=df['States/ Union Territories']       #store all the state and UT names in x 

y=df[["Total - 2001","Total - 2011"]]   #store all the values of Total-2001 and Total-2011 in y
#plot the graph of total litracy rate across the nation

plt.figure(figsize=(60,20))

plt.plot(x,y)

plt.title("Total Literacy Rate Across Nation")

plt.legend(["Total-2001","Total-2011"],loc='upper left')

plt.style.use('ggplot')
#sort the values of total-2001 column and take upper 5 rows

lowest_2001=df.sort_values('Total - 2001',na_position='first').head() 

lowest_2001
#sort the values of total-2001 column and take lower 5 rows

highest_2001=df.sort_values('Total - 2001',na_position='first').tail()
highest_2001
#plot the graph to check lowest and highest Total literacy rate in 2001

plt.figure(figsize=(20,7))

x_t_2001=lowest_2001["States/ Union Territories"]

y_t_2001=lowest_2001["Total - 2001"]

plt.plot(x_t_2001,y_t_2001,marker='o',label='Lowest_2001')

x_h_2001=highest_2001["States/ Union Territories"]

y_h_2001=highest_2001["Total - 2001"]

plt.plot(x_h_2001,y_h_2001,marker='o',label='Highest_2001')

plt.title('Lowest and Highest "Total Literacy" rate in 2001')

plt.legend(loc='upper left')
##sort the values of total-2011 column and take upper 5 rows

lowest_2011=df.sort_values('Total - 2011',na_position='first').head()

lowest_2011
#sort the values of total-2011 column and take lower 5 rows

highest_2011=df.sort_values('Total - 2011',na_position='first').tail()
highest_2011
plt.figure(figsize=(15,7))

x_t_2011=lowest_2011["States/ Union Territories"]

y_t_2011=lowest_2011["Total - 2011"]

plt.plot(x_t_2011,y_t_2011,marker='o',label="Lowest_2011")

x_h_2011=highest_2011["States/ Union Territories"]

y_h_2011=highest_2011["Total - 2011"]

plt.plot(x_h_2011,y_h_2011,marker='o',label="Highest_2011")

plt.title('Lowest and Highest "Total Literacy" rate in 2001')

plt.legend(loc='upper left')
#sort the values of total-per.change column

tpc=df.sort_values('Total - Per. Change')
tpc.head()
#plot the graph to check  percentage change in Total literacay rate during 2001-2011

plt.figure(figsize=(50,20))

xtpc=tpc["States/ Union Territories"]

ytpc=tpc["Total - Per. Change"]

plt.bar(xtpc,ytpc,color='maroon')

plt.title('Total Per Change')

plt.xlabel('States/Union Terrirories')

plt.ylabel('Total Per Change')
#sort the values by Rural-2001 column

df.sort_values('Rural - 2001',inplace=True)
df.head()
plt.figure(figsize=(50,20))

x1=df["States/ Union Territories"]         #store all the state and UT names in x1

y1=df[["Rural - 2001","Rural - 2011"]]     #store all the values of Rural-2001 and Rural-2011 in y1

plt.plot(x1,y1)

plt.title("Rural Literacy Rate Across Nation")

plt.legend(["Rural-2001","Rural-2011"],loc='upper left')

#sort the values of Rural-2001 column and take upper 5 rows

lowest_r_2001=df.sort_values('Rural - 2001',na_position='first').head()
lowest_r_2001
#sort the values of Rural-2001 column and take lower 5 rows

highest_r_2001=df.sort_values('Rural - 2001',na_position='first').tail()
highest_r_2001
#plot the graph to check Lowest and Highest Rural Literacy rate in 2001

plt.figure(figsize=(15,7))

x_r_2001=lowest_r_2001["States/ Union Territories"]

y_r_2001=lowest_r_2001["Rural - 2001"]

plt.plot(x_r_2001,y_r_2001,marker='o',label="Lowest_2001")

x_ru_2001=highest_r_2001["States/ Union Territories"]

y_ru_2001=highest_r_2001["Rural - 2001"]

plt.plot(x_ru_2001,y_ru_2001,marker='o',label="Highest_2001")

plt.title('Lowest and Highest "Rural Literacy" rate in 2001')

plt.legend(loc='upper left')
#sort the values of Rural-2011 column and take upper 5 rows

lowest_r_2011=df.sort_values('Rural - 2011',na_position='first').head()
lowest_r_2011
#sort the values of Rural-2011 column and take lower 5 rows

highest_r_2011=df.sort_values('Rural - 2011',na_position='first').tail()
highest_r_2011
#plot the graph to check the Lowest and highest Rural literacy rate in 2011

plt.figure(figsize=(15,7))

x_r_2011=lowest_r_2011["States/ Union Territories"]

y_r_2011=lowest_r_2011["Rural - 2011"]

plt.plot(x_r_2011,y_r_2011,marker='o',label="Lowest_2011")

x_ru_2011=highest_r_2011["States/ Union Territories"]

y_ru_2011=highest_r_2011["Rural - 2011"]

plt.plot(x_ru_2011,y_ru_2011,marker='o',label="Highest_2011")

plt.title('Lowest and Highest "Rural Literacy" rate in 2011')

plt.legend(loc='upper left')
#sort the values of Rural-per.change column

rpc=df.sort_values('Rural - Per. Change')
rpc.head(3)
#plot the graph to check percentage change in Rural Literacy Rate during 2001-2011

plt.figure(figsize=(60,20))

xrpc=rpc['States/ Union Territories']

yrpc=rpc['Rural - Per. Change']

plt.bar(xrpc,yrpc)

#plt.title('Rural Per Change')

plt.xlabel('State/Union Territories')

plt.ylabel('Rural Per Change')
df.sort_values('Urban - 2001',inplace=True)  #sort the values by Urban-2001 column
df.head(3)
plt.figure(figsize=(60,20))

x2=df['States/ Union Territories']            #store the States and UT names in x2

y2=df[["Urban - 2001","Urban - 2011"]]        #store all the values of Urban-2001 and urban-2011 in y2

plt.plot(x2,y2)

plt.title('Urban Literacy Rate Across Nation')

plt.legend(["Urban-2001","Urban-2011"],loc='upper left')
#sort the values of Urban-2001 column and take upper 5 rows

lowest_u_2001=df.sort_values('Urban - 2001',na_position='first').head()
lowest_u_2001
#sort the values of Urban-2001 column and take lower 5 rows

highest_u_2001=df.sort_values('Urban - 2001',na_position='first').tail()
highest_u_2001
#plot the graph to check the Lowest an Highest Urban Literacy rate in 2001

plt.figure(figsize=(15,7))

x_u_2001=lowest_u_2001["States/ Union Territories"]

y_u_2001=lowest_u_2001["Urban - 2001"]

plt.plot(x_u_2001,y_u_2001,marker='o',label="Lowest_2001")

x_ub_2001=highest_u_2001["States/ Union Territories"]

y_ub_2001=highest_u_2001["Urban - 2001"]

plt.plot(x_ub_2001,y_ub_2001,marker='o',label="Highest_2001")

plt.title('Lowest And Highest Urban Literacy in 2001')

plt.legend(loc='upper left')
#sort the values of Urban-2011 column and take upper 5 rows

lowest_u_2011=df.sort_values('Urban - 2011',na_position='first').head()
lowest_u_2011
#sort the values of Urban-2011 column and take lower 5 rows

highest_u_2011=df.sort_values('Urban - 2011',na_position='first').tail()
highest_u_2011
#plot the graph to check Lowest and Highest Urban Literacy rate in 2011

plt.figure(figsize=(15,7))

x_u_2011=lowest_u_2011["States/ Union Territories"]

y_u_2011=lowest_u_2011["Urban - 2011"]

plt.plot(x_u_2011,y_u_2011,marker='o',label="Lowest_2011")

x_ur_2011=highest_u_2011["States/ Union Territories"]

y_ur_2011=highest_u_2011["Urban - 2011"]

plt.plot(x_ur_2011,y_ur_2011,marker='o',label="Highest_2011")

plt.title('Lowest and Highest Urban Literacy rate in 2011')

plt.legend(loc='upper left')
#sort the values of Urben-per.change

upc=df.sort_values('Urban - Per. Change')
upc.head(3)
#plot the graph to check percentage change in 'Urban Literacy Rate' during 2001-2011

plt.figure(figsize=(60,20))

xupc=upc['States/ Union Territories']

yupc=upc['Urban - Per. Change']

plt.bar(xupc,yupc,color='darkblue')

plt.xlabel('States/ Union Territories')

plt.ylabel('Urban - Per. Change')
Total_2001 = df.groupby("Category")["Total - 2001"].mean().reset_index()

Total_2011 = df.groupby("Category")["Total - 2011"].mean().reset_index()





Rural_2001 = df.groupby("Category")["Rural - 2001"].mean().reset_index()

Rural_2011= df.groupby("Category")["Rural - 2011"].mean().reset_index()





Urban_2001= df.groupby("Category")["Urban - 2001"].mean().reset_index()

Urban_2011= df.groupby("Category")["Urban - 2011"].mean().reset_index()





plt.figure(1)

Total_2001.plot(kind='barh',color=[['green','blue']],figsize=(5,3))

plt.ylabel('States           UT')

Total_2011.plot(kind='barh',color=[['green','blue']],figsize=(5,3))

plt.ylabel('States           UT')



plt.figure(2)

Rural_2001.plot(kind='barh',color=[['green','blue']],figsize=(5,3)) 

plt.ylabel('States           UT')

Rural_2011.plot(kind='barh',color=[['green','blue']],figsize=(5,3))

plt.ylabel('States           UT')

plt.figure(3)

Urban_2001.plot(kind='barh',color=[['green','blue']],figsize=(5,3)) 

plt.ylabel('States           UT')

Urban_2011.plot(kind='barh',color=[['green','blue']],figsize=(5,3))

plt.ylabel('States           UT')