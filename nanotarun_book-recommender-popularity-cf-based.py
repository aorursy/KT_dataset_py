import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import missingno

import warnings 

warnings.filterwarnings("ignore")



pd.set_option("display.max_rows",None)

pd.set_option("display.max_columns",None)
# System

import warnings

import os

warnings.filterwarnings("ignore")

%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
user=pd.read_csv("/kaggle/input/bookcrossing/bx-csv-dump/BX-Users.csv",error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1')

books=pd.read_csv("/kaggle/input/bookcrossing/bx-csv-dump/BX-Books.csv",error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1')

ratings=pd.read_csv("/kaggle/input/bookcrossing/bx-csv-dump/BX-Book-Ratings.csv",error_bad_lines=False, delimiter=';', encoding = 'ISO-8859-1')
user.head()
books.head()
ratings.head()
data = pd.merge(books, ratings, on='ISBN', how='left')

data.head()
data=pd.merge(data, user, on='User-ID', how='left')
data.head()
data.drop(["Image-URL-S","Image-URL-M","Image-URL-L"],axis=1,inplace=True)
data.head()
#missingno.matrix(data)
data.shape
data["User-ID"].unique().shape
data[data.ISBN=="034545104X"].head()
# For simplification of the problem we gonna drop the age column too



data.drop("Age",axis=1,inplace=True)
data.head()
data["Year-Of-Publication"].unique()
data[data["Year-Of-Publication"]=="DK Publishing Inc"]
data["Year-Of-Publication"].replace({"DK Publishing Inc":2000},inplace=True)
data["Year-Of-Publication"].unique()
data[data["Year-Of-Publication"]=="Gallimard"]
data["Year-Of-Publication"].replace({"Gallimard":2003},inplace=True)
data[data["Year-Of-Publication"]==0].head()
data["Year-Of-Publication"].mode()
data["Year-Of-Publication"].replace({0:2002},inplace=True)
data["Year-Of-Publication"].unique()
data["Year-Of-Publication"]=data["Year-Of-Publication"].astype(int)
list1=[]

for i in data["Year-Of-Publication"]:

    if i >2016:

        i=2016

    list1.append(i)

#out = np.where(data.values <= q_05,q_05, np.where(data >= q_95, q_95, data))

for i in list1:

    if i>2016:

        print(i)
data["Year-Of-Publication"]=list1
data["Year-Of-Publication"].value_counts().sort_index().head()
list2=[]

for i in data["Year-Of-Publication"]:

    if i <1376:

        i=1376

    list2.append(i)
data["Year-Of-Publication"]=list2
data["Year-Of-Publication"].unique()
data.head()
data.Publisher.isnull().sum()
data.Publisher.unique()
data["Publisher"].sort_values().head()
data[data.Publisher.isnull()]
data.Publisher.iloc[824289]="Editions P. Terrail"
data.Publisher.iloc[824598]="Editions P. Terrail"
data.head()
data.isnull().sum()
data[data["Book-Author"].isnull()]
data["Book-Author"].value_counts().sort_values(ascending=False).head()
data["Book-Author"].iloc[929219]="Stephen King"
data["Book-Author"].isnull().sum()
data["Book-Author"].head()
# Droping the location column
data.drop("Location",axis=1,inplace=True)
data.head()
data["User-ID"].isnull().sum()
data.isnull().sum()
data.dropna(inplace=True)
data.isnull().sum()
data.shape
data.head()
sns.countplot(data["Book-Rating"])
data["Book-Rating"].value_counts()
data.head()
ratng=data.copy()
ratng["Book-Rating"].replace(0.0,None,inplace=True)
ratng.head()
ratng[ratng["Book-Rating"]==0]
ratng["Book-Rating"].iloc[0]=5
### lets check the distribution again
sns.countplot(ratng["Book-Rating"])
rec1=pd.DataFrame(ratng.groupby(["ISBN","Book-Title","Book-Author"])["Book-Rating"].sum().sort_values(ascending=False).head(10))
rec1
top_index=ratng["User-ID"].value_counts().sort_values(ascending=False).head(500).index

top_index
df=ratng[ratng["User-ID"].isin(top_index)]

df.head()
df.head()
idcount=df["User-ID"].value_counts()
idcount.shape
df121=df[df["User-ID"].isin(idcount[idcount>=1500].index)]
zxc=df121.groupby("Book-Title")["Book-Rating"].sum().reset_index()

zxc.head()
zxc=zxc[zxc["Book-Rating"]>200]
zxc["Book-Rating"].max()
df_mat=df121[df121["Book-Title"].isin(zxc["Book-Title"])]
matrix=df_mat.pivot(index="User-ID",columns="ISBN",values="Book-Rating")
matrix.fillna(0,inplace=True)
matrix
from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity(matrix)

np.fill_diagonal(cos_sim,0)        # zero here means that both ids are same,it should be 1 here but i am using 0 so as to ease further coding process

rec_cos=pd.DataFrame(cos_sim,index=matrix.index)

rec_cos.columns=matrix.index

rec_cos.head()
df_mat[df_mat["User-ID"]==16795.0][["Book-Title","Book-Rating"]].head()
df_mat[df_mat["User-ID"]==135149.0][["Book-Title","Book-Rating"]].head()
def sim(userid,n):          # userid is the id for which recommendations has to be made, n represents total no. of similiar users wanted 

    print(np.array(rec_cos[userid].sort_values(ascending=False).head(n).index))
print(np.array(rec_cos[98391.0].sort_values(ascending=False).head(10).index))
sim(98391.0,20)        # .0 has to be added in front of every id as it is working column wise instead of row wise
def book_recommender():              # userid is the id for which recommendations has to be made, n represents total no. of similiar users wanted 

    print()

    print()

    userid = int(input("Enter the user id to whom you want to recommend : "))

    print()

    print()

    n= int(input("Enter how many books you want to recommend : "))

    print()

    print()

    arr=np.array(rec_cos[userid].sort_values(ascending=False).head(5).index)

    recom_arr=[]



    for i in arr:

        recom_arr.append(df_mat[df_mat["User-ID"]==i][["Book-Title","Book-Rating"]].sort_values(by="Book-Rating",ascending=False))

    

    return(pd.Series(recom_arr[0].append([recom_arr[1],recom_arr[2],recom_arr[3],recom_arr[4]]).groupby("Book-Title")["Book-Rating"].mean().sort_values(ascending=False).index).head(n))
book_recommender()