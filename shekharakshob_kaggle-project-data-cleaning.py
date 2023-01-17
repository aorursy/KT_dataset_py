import numpy as np

import pandas as pd

import re

from sklearn.preprocessing import StandardScaler
df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
df.head(60)
df.shape





# Removing duplicats
df.drop_duplicates(keep = False, inplace = True)
df.shape
df.info() 



# Dropping price column as 93% of the data is 0 which correlated with Free in type column
df.drop(['Price'],axis=1,inplace=True) 







# changing object to numberic value


df["Installs"] = df["Installs"].replace({"\W+":""},regex=True).replace({"\D+":""},regex=True).replace({"\s+":""},regex=True).replace("",np.nan).astype("float64") 

df["Size"] = df["Size"].replace({"[^\d\.']":""},regex=True).replace("",np.nan).astype("float32")*df["Size"].replace({"[^mMkK']":""},regex=True).replace({"M":1000000,"k":1000,"\s+":""}).replace("",1).astype("float64")

df["Reviews"] = df["Reviews"].replace({"[^\d\.']":""},regex=True).replace("",np.nan).astype("float32")*df["Reviews"].replace({"[^mMkK']":""},regex=True).replace({"M":1000000,"k":1000,"\s+":""}).replace("",1).astype("float64")
df.head(60)





# % of empty value in each column
for k in df.columns:

    print(k,round((df[k].isnull().sum()/df[k].count()*100),3))





df.Size.fillna(df.Size.mean(),inplace=True)

df.Rating.fillna(df.Rating.mode()[0],inplace=True)
#removing rest of the nan data row 

for k in df.columns:

    df.dropna(subset = [k], inplace=True)
for k in df.columns:

    print(k,round((df[k].isnull().sum()/df[k].count()*100),3))
df.select_dtypes(include=['float64']).columns
import seaborn as sns

import matplotlib.pyplot as plt





# Removing outliers 





#removing outliers 

lower_bound = 0.1

upper_bound = 0.95

# there are lower bound and upper bound vlaues of the percentile 

res = df.Rating.quantile([0.1,0.95])

df1 = df[(df.Rating > res.values[0]) & (df.Rating < res.values[1])] 

df1.shape