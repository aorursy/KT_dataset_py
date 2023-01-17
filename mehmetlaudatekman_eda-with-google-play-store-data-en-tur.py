# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



import scipy.stats as stats



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")

#Importing the data
data.head() #It is showing us first 5 rows 
data.tail() #It is showing us last 5 rows
data.info
"""

First,I am going to take a look at the features of data

"""

data.columns

data.head()

#Cleaning Price Data



price = [i for i in data["Price"]]



def currency_cleaner(currency_li):

    """

    This function cleans price data

    """

    cleaned_data = []

    for prc in currency_li:

        if "$" in prc:

            prc = prc.replace("$","")

        cleaned_data.append(prc)

    return cleaned_data





     

data["Price"] = currency_cleaner(price)



""" While I clean the data, I see this row. That is interesting."""

data[data["Price"]=="Everyone"]

data.drop(index=10472,inplace=True)

data.reset_index(inplace=True)



data["Price"] = data["Price"].astype(float) #Category =>> Float





#Cleaning Reviews Data



data["Reviews"] = data["Reviews"].astype(float) #Category =>> Float





#Cleaning Size Data

size = [i for i in data["Size"]]

def size_cleaner(size_li):

    """

    This function converts kilobytes to megabytes 

    """

    clean_li = []

    for sz in size_li:

        

        if "M" in sz:

            sz = sz.replace("M","")

            sz = float(sz)

        

        elif "k" in sz:

            sz = sz.replace("k","")

            sz = float(sz)

            sz = sz/1024 #Because one megabyte contains 1024 kilobytes

        

        elif "Varies with device" in sz:

            continue                        

        clean_li.append(sz)

    return clean_li







ind = data[data["Size"]=="Varies with device"].index #I have to drop the rows that have this value 

data.drop(axis=0,inplace=True,index=ind)             #Because I cant convert them to numerical values



size = size_cleaner(size)

data["Size"] = size

data.Size

data["Size"] = data["Size"].astype(float)



#Cleaning Install Data



#install_data = [i for i in data["Install"]]

def install_cleaner(install_li):

    clean_list = []

    for i in install_li:        

        if "," in i:

            i = i.replace(",","")

        if "+" in i:

            i = i.replace("+","")

        i = int(i)

        clean_list.append(i)

    return clean_list





data["Installs"] = install_cleaner(list(data["Installs"]))



#Cleaning is over!
data.corr()
fig,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),ax=ax,annot=True)

plt.title("Correlation Between Features ")

plt.show()
corr = data.corr()

corr
#And I am going to drop index because I will not use this

corr.drop("index",axis = 1,inplace=True)

corr
g = sns.jointplot("Installs","Reviews",data=data,kind="scatter",size=6)

g.annotate(stats.pearsonr)

plt.subplots_adjust(top=0.9)

plt.suptitle("Correlation Between Installs and Reviews\n \t Joint Plot \t", fontsize = 16)

plt.show()



g2 = sns.lmplot("Installs","Reviews",data=data,size=6)

plt.subplots_adjust(top=0.9)

plt.suptitle("Correlation Between Installs and Reviews\n \t LM Plot \t ", fontsize = 16)

plt.show()

category = data.groupby(data["Category"]).mean()

category.drop("index",axis=1,inplace=True)

category
def create_barplot(ftr_names,dataBar):

   

    for ftr in ftr_names:      

        fig,ax = plt.subplots(figsize=(10,10))

        new_index = dataBar[ftr].sort_values(ascending=False).index.values

        new_data = dataBar.reindex(new_index)

        sns.barplot(new_data.index.values,list(new_data[ftr]),ax=ax,order=new_data.index)

        plt.xlabel("Index")

        plt.ylabel(ftr)

        plt.xticks(rotation=90)

        plt.show()

create_barplot(category.columns,category)
rating = data.groupby("Rating").mean()

rating.drop("index",axis=1,inplace=True)

rating.index = rating.index.astype(str)

rating

create_barplot(rating.columns,rating)
app_size = data.groupby("Size").mean()

app_size.drop("index",axis=1)

app_size



size_list = data["Size"].value_counts()

sz = dict(size_list)

sz = list(sz.keys())

sz = sz[0:30]

sizedt = data.groupby("Size").mean()



for i in sizedt.index:

    

    if not i in sz:

        sizedt.drop(index=i,inplace=True)



sizedt.drop("index",axis=1)
create_barplot(sizedt.columns,sizedt)
review_count = data.groupby("Reviews").mean()

review_count.drop("index",axis=1,inplace=True)
review_list = data["Reviews"].value_counts()

rl = dict(review_list)

rl = list(rl.keys())

rl = rl[0:30]



for i in review_count.index:

    

    if not i in sz:

        review_count.drop(index=i,inplace=True)



review_count    

create_barplot(review_count.columns,review_count)
data["Rating"].fillna(0.0,inplace=True)

install_count = data.groupby("Installs").mean()

install_count.drop("index",axis=1,inplace=True)

install_count
create_barplot(install_count.columns,install_count)
app_type = data.groupby("Type").mean()

app_type.drop("index",axis=1,inplace=True)



create_barplot(app_type.columns,app_type)
category_names = [i for i in data["Category"]]

data2 = data.copy()

ind =  data2["Installs"].sort_values(ascending=False).index.values

data2 = data2.reindex(ind)

data2.head()

def determine3apps_uniquely(category,data):

    categorized_apps = data[data["Category"]==category]["App"].unique()

    app_list = list(categorized_apps)[:3]

    return app_list



def create_dataframe(appnames):

    return_dataFrame = pd.DataFrame({"level_0":[],"index":[],"App":[],"Category":[],"Rating":[],

                                    "Reviews":[],"Size":[],"Installs":[],"Type":[],"Price":[],

                                    "Content Rating":[],"Genres":[],"Last Updated":[],"Current Ver":[],"Android Ver":[]})

    for app in appnames:

        dataApp = data2[data2["App"]==app]

        dataApp.reset_index(inplace=True)

        dataApp = dataApp[dataApp.index == 0]

        return_dataFrame = pd.concat([return_dataFrame,dataApp],axis=0)

        return_dataFrame.drop("level_0",axis=1,inplace=True)

        #We know, some apps have copies more than one in data, so I'll take only first row!

    return return_dataFrame



game_apps = determine3apps_uniquely("GAME",data2)

game_df = create_dataframe(game_apps)

game_df
def create_pointPlot(df):

    """

    This function draws point plots!

    """

    df.drop("index",axis=1,inplace=True)

    #I'll use describe columns because I am only wanting numerical features

    features = df.describe().columns 

    

    for ftr in features:

        fig,ax  = plt.subplots(figsize=(12,6))

        sns.pointplot(df["App"],df[ftr])

        plt.xlabel("APP NAMES")

        plt.ylabel(ftr.upper())

        plt.xticks(rotation = 90)

        plt.title("COMPARING APPS IN TERMS OF "+ftr.upper())

        plt.show()

    
create_pointPlot(game_df)
#App Detecting!

tools_apps = determine3apps_uniquely("TOOLS",data2)

tools_df = create_dataframe(tools_apps)



tools_df
#Comparing and visualizing with using point plot from Seaborn library



create_pointPlot(tools_df)
unique_android_data = data["Android Ver"].unique()

unique_android_data
#Dropping NaN values



data.dropna(subset=["Android Ver"],inplace=True)

def android_data_cleaner(dataFrame):

    

    androidData = list(dataFrame["Android Ver"])

    cleanData = []

    for aD in androidData:

        

        if "and up" in aD:

            aD = aD.replace("and up","")

        

        if "Varies with device" in aD:

            aD = "0.0"

        

        if "-" in aD:

            list1 = aD.split("-")

            aD = list1[0]            

        

        if "4.4W" in aD: 

            aD = "4.4"

        

        list1 = aD.split(".")

        if len(list1) == 3:

             aD = list1[0]+"."+list1[1]+list1[2]

        

        cleanData.append(aD)

        

    return cleanData





clean_data = android_data_cleaner(data)

data["Android Ver"] = clean_data



data["Android Ver"] = data["Android Ver"].astype(float)

data.head()
data.dtypes
def piechartCreator(dF,feature,figsize=(8,8)):

    """

    A function that creates pie charts 

    """

    value_counts = dF[feature].value_counts()

    fig,ax = plt.subplots(figsize=figsize)

    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',shadow=True)

    plt.title("Comparing Apps by "+feature,fontsize=14,color = "Purple")

    plt.show()

    

piechartCreator(data,"Type")                        

    
piechartCreator(data,"Category",figsize=(17,17))
piechartCreator(data,"Installs",figsize=(20,20))
def outlier_dropper(dF):

    dF = dF.drop("index",axis=1)

    dF = dF.drop("Price",axis=1)

    """

    I have to drop Price data because most of the values are 0 and It might create problems in the future

    """

    ftrSteps = []

    

    

    descrb = dF.describe()

    

    ftrQ1 = [i for i in descrb.loc["25%"]]

    ftrQ3 = [i for i in descrb.loc["75%"]]

    for ftr in descrb:

        IQR = descrb[ftr]["75%"] - descrb[ftr]["25%"]

        IQR = IQR * 1.5

        ftrSteps.append(IQR)

    

    drop_counter = 0

    outlier_counter = 0

    dFcolumns = ["Rating","Reviews","Size","Installs","Android Ver"]

    for r in range(0,len(dF)):

        

        row = dF[dF.index == r]

        for i in range(0,len(dFcolumns)) :

            if list(row[dFcolumns[i]].values) != []:

                if float(row[dFcolumns[i]].values) < float(ftrQ1[i]) -  float(ftrSteps[i]) or float(row[dFcolumns[i]].values) > float(ftrQ1[i]) +  float(ftrSteps[i]):

                    outlier_counter +=1

                    if outlier_counter == 2:

                        outlier_counter = 0

                        drop_counter += 1

                 

                        dF.drop(r,inplace=True)

                        break

                

    dF.reset_index(inplace=True)

    return drop_counter,dF

                
#First I'll crate a new copy of data



nwData = data.copy()



#Then, I'll drop NaN values

nwData.dropna(inplace=True)



#And now, We are ready to start our function

drop_count,nwData = outlier_dropper(nwData)



print(f" {drop_count} rows dropped! ")



nwData.head()