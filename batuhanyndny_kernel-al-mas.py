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



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/BlackFriday.csv")

data.info()
# spliting dataframe to cities

print(data["City_Category"].unique())

dict_of_city = {k: v for k, v in data.groupby('City_Category')}

cityA =dict_of_city["A"]

cityB =dict_of_city["B"]

cityC=dict_of_city["C"]

#splitting cities to male and female 

cityAfm = {k: v for k , v in cityA.groupby("Gender")}

cityBfm = {k: v for k , v in cityB.groupby("Gender")}

cityCfm = {k: v for k , v in cityC.groupby("Gender")}

cityA_males = cityAfm["M"]

cityA_females = cityAfm["F"]

cityB_males = cityBfm["M"]

cityB_females = cityBfm["F"]

cityC_males = cityCfm["M"]

cityC_females = cityCfm["F"]

# Creating dataframe that shows buyers gender

dfgender = pd.DataFrame(data["Gender"])

dist_of_gender=dfgender["Gender"].value_counts()

dogdf=pd.DataFrame(dist_of_gender,columns=["Gender"])

print(dogdf.T)

# ploting dataframe

dogdf.T.plot(kind="bar",figsize=(8,8))

plt.show()
#the correlation map that shows what males buy at city A 

f , ax = plt.subplots(figsize= (5,5))

sns.heatmap(cityA_males.corr() ,annot=True ,linewidths = .3, fmt= ".3f", ax =ax)

plt.show()
# getting a little info about our data set

data.head()

#calculating city purchases

cityA_purchase=cityA["Purchase"].sum()

cityB_purchase=cityB["Purchase"].sum()

cityC_purchase=cityC["Purchase"].sum()

purchases_df=pd.DataFrame({"cityA":[cityA_purchase],"cityB":[cityB_purchase],"cityC":[cityC_purchase]})
#ploting city purchases for general knowledge

purchases_df.plot(kind="bar")

plt.legend()

plt.yticks()

plt.ylabel("Money")

plt.xlabel("Cities")

plt.title("City-Purchase")

plt.show()
#List Comprehension exercise 

#l1= [5,10,12,15,20,91,2,826,28957,181,563,829]

#l2= [i for i in l1 if i >100]

#l3 = [i for i in l1 if i < 500]

#l3.sort()

#print("L3",l3)
# I take the material status 0 as bad and 1 as good for products

data["Quality"] = ["Good" if i==1 else "Bad" for i in data.Marital_Status] 

data.columns
goods_quality=data.drop(["User_ID","Gender","Age","Occupation","City_Category","Stay_In_Current_City_Years","Marital_Status","Product_Category_1","Product_Category_2","Product_Category_3","Purchase"],axis=1)

print(goods_quality.columns)

goods_quality.head()
# Ploting the quality of goods 

qualdf=pd.DataFrame(goods_quality["Quality"])

counted=qualdf["Quality"].value_counts()

qualities_df=pd.DataFrame(counted)

qualities_df.T.plot(kind="bar")

plt.show()
print(data.Age.unique())
#data["deneme"]=[for i in data.Age "Childs and Teens" if i == "0-17" else "Early Adult" if i == "18-25" else "Adult" if i == "26-35" else "Midlife" if i == "36-45" else "Mature Adult" if i == "46-50" else "Late Adult" if i == "51-55" else "Eldery" if i == "55+" else ]
#data["Stage"]=["Childs and Teens" if i == "0-17" else "Early Adult" if i == "18-25" else "Adult" if i == "26-35" else "Midlife" if i == "36-45" else "Mature Adult" if i == "46-50" else "Late Adult" if i == "51-55" else "Eldery" if i == "55+" for i in data.Age]

Ages=[]

for i in data.Age:

    if i == "0-17":

        Ages.append("Childs and Teens")

    elif i== "18-25":

        Ages.append("Early Adult")

    elif i== "26-35":

        Ages.append("Adult")

    elif i== "36-45":

        Ages.append("Midlife")

    elif i== "46-50":

        Ages.append("Mature Adult")

    elif i== "51-55":

        Ages.append("Late Adult")

    elif i== "55+":

        Ages.append("Eldery")

    else:

        pass

ages_df=pd.DataFrame(Ages,columns=["Stage"])

age_count=ages_df["Stage"].value_counts()

age_count.plot(kind="bar")

plt.title("Customers")

plt.show()



        

    
data.head()
#Ploting Stages with gender

gender_age=pd.concat([dfgender,ages_df],axis=1)

dict_of_gender = {k: v for k, v in gender_age.groupby("Gender")}

F_age=pd.DataFrame(dict_of_gender["F"])

M_age=pd.DataFrame(dict_of_gender["M"])

f_counts=F_age["Stage"].value_counts()

m_counts=M_age["Stage"].value_counts()

dff= pd.DataFrame(f_counts)

dff.rename(columns={"Stage":"Female"},inplace=True)

Female_stage=dff.T

dff2=pd.DataFrame(m_counts)

dff2.rename(columns={"Stage":"Male"},inplace=True)

Male_stage=dff2.T

gender_stage= pd.concat([Female_stage,Male_stage],axis=0,ignore_index=False,sort=False)

gender_stage.plot(kind="bar",figsize=(5,5))

plt.show()

#Let's define a  null finder

def nullfinder(x):

    a=x.isnull()

    y=a.any(axis=0)

    z=pd.DataFrame(y)

    nulls=z[z.isin([True]).any(axis=1)]

    return("Null values found in: ",list(nulls.index.values))
# Outputs a list of columns that contains null values

nullfinder(data)
#define a null checker

def nullcheck(x): 

    try:

        assert x.notnull().all()

    except AssertionError:

            print("Null values found" )

    return  print("Check Completed")
nullcheck(data.Product_Category_3)
#filling empty datas

data.fillna("Unknown",inplace=True)
nullcheck(data.Product_Category_3)
data.head()
data = pd.read_csv("../input/BlackFriday.csv")
data.head()
df1=data.set_index(["User_ID","Gender"])
df1.index=data.index
df1["Gender"]=data.Gender

df1["User_ID"]=data.User_ID
df1