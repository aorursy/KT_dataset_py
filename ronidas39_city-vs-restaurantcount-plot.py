import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

from glob import glob as gb
#list all the directories

dirs=os.listdir("../input/zomato_data/")

dirs
len(dirs)
#storing all the files from every directory

li=[]

for dir1 in dirs:

    files=os.listdir(r"../input/zomato_data/"+dir1)

    #reading each file from list of files from previous step and creating pandas data fame    

    for file in files:

        

        df_file=pd.read_csv("../input/zomato_data/"+dir1+"/"+file,quotechar='"',delimiter="|")

#appending the dataframe into a list

        li.append(df_file.values)

    

    
len(li)
#numpys vstack method to append all the datafames to stack the sequence of input vertically to make a single array

df_np=np.vstack(li)
#no of rows is represents the total no restaurants ,now of coloumns(12) is columns for the dataframe

df_np.shape
#creating final dataframe from the numpy array

df_final=pd.DataFrame(df_np)
#adding the header columns

df_final=pd.DataFrame(df_final.values, columns =["NAME","PRICE","CUSINE_CATEGORY","CITY","REGION","URL","PAGE NO","CUSINE TYPE","TIMING","RATING_TYPE","RATING","VOTES"])

#displaying the dataframe

df_final
#header column "PAGE NO" is not required ,i used it while scraping the data from zomato to do some sort of validation,lets remove the column

df_final.drop(columns=["PAGE NO"],axis=1,inplace=True)
#display the dataframe again

df_final
#lets count how many unique cities are there 



df_final["CITY"].unique()
len(df_final["CITY"].unique())
#lets check city wise restaurant counts and save it in ascending order



city_vs_count=df_final["CITY"].value_counts().sort_values(ascending=True)
city_vs_count
#lets check max count

count_max=max(city_vs_count)
#lets find for city count is max



for x,y in city_vs_count.items():

    if(y==count_max):

        print(x)

    
#lets find for city count is min



min_count=min(city_vs_count)



for x,y in city_vs_count.items():

    if(y==min_count):

        print(x)
#lets plot citywise restaurant count in barh form



fig=plt.figure(figsize=(20,40))

city_vs_count.plot(kind="barh",fontsize=30)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.ylabel("city names",fontsize=50,color="red",fontweight='bold')

plt.title("CITY VS RESTAURANT COUNT GRAPH",fontsize=50,color="BLUE",fontweight='bold')
#lets plot citywise restaurant count in barh form,and each bar should display the count of the corresponding restuants for that city



fig=plt.figure(figsize=(20,40))

city_vs_count.plot(kind="barh",fontsize=30)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.ylabel("city names",fontsize=50,color="red",fontweight='bold')

plt.title("CITY VS RESTAURANT COUNT GRAPH",fontsize=50,color="BLUE",fontweight='bold')

for v in range(len(city_vs_count)):

    #plt.text(x axis location ,y axis location ,text value ,other parameters......)

    plt.text(v+city_vs_count[v],v,city_vs_count[v],fontsize=20,color="BLUE",fontweight='bold')
#THATS ALL GUYS SEE YOU IN THE NEXT KERNEL