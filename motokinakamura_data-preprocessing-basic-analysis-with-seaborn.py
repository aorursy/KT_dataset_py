#import libralies
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
marvel = pd.read_csv("../input/marvel-wikia-data.csv")
marvel.shape
#See the loss of data in each colmns
print(marvel.shape)
print("----------------------")
print(marvel.isnull().any())
print("----------------------")
print(marvel.isnull().sum())
#drop useless columns
marvel = marvel.drop(["GSM"],axis=1)
# fill loss cells
marvel.ID = marvel.ID.fillna("unknown")
marvel.ALIGN = marvel.ALIGN.fillna("unknown")
marvel.EYE = marvel.EYE.fillna("unknown")
marvel.HAIR = marvel.HAIR.fillna("unknown")
marvel.SEX = marvel.SEX.fillna("unknown")
marvel.ALIVE = marvel.ALIVE.fillna("unknown")

#change name of column
marvel = marvel.rename(columns= {
    "FIRST APPEARANCE" : "FIRST_APPEARANCE"
    })
#See the loss of data in each colmns
print(marvel.shape)
print("----------------------")
print(marvel.isnull().any())
print("----------------------")
print(marvel.isnull().sum())
#remove rows which has "Nan"
marvel = marvel.dropna()

marvel.shape
plt.style.use('ggplot') 

#count data of SEX
plt.figure(figsize=(15, 5))
sns.countplot(x=marvel.SEX,
              data=marvel,
              palette="Pastel2" ,
              order=["Male Characters","Female Characters","Agender Characters","Genderfluid CharactersSEX","unknown"])
#count data of ["SEX"] on ["Year"]
marvel.groupby("Year")["SEX"].value_counts().unstack().plot(figsize=(15,5))
#what about ID
plt.figure(figsize=(15, 5))
sns.countplot(x=marvel.ID, data=marvel, palette="Pastel1") 
#count data of ["ID"] on ["Year"]
marvel.groupby("Year")["ID"].value_counts().unstack().plot(figsize=(15,8))
#what about EYE
plt.figure(figsize=(15, 8))
sns.countplot(x=marvel.EYE, data=marvel, palette="Pastel1") 
#what about EYE without "unknown"
remove_EYE = marvel[marvel.EYE != "unknown"]

plt.figure(figsize=(30, 10))
sns.countplot(x=remove_EYE.EYE, data=remove_EYE, palette="Pastel1")
#count data of ["ALIVE"] on ["Year"]
marvel.groupby("Year")["ALIVE"].value_counts().unstack().plot(figsize=(15,10))