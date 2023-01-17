# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Pokemon.csv")
print("description of data:\n",df.describe())
print("shape of data:\n",df.shape)
print("datatypes:\n",df.dtypes)
print("null values in dataset:\n",df.isnull().sum())
df.drop(columns="#",inplace=True) # droping the unnecessary column
df.rename(columns = {"Type 1" : "Type_1",
          "Type 2": "Type_2",
          "Sp. Atk" : "Sp_Atk",
          "Sp. Def" : "Sp_Def"},inplace=True) # renaming the column names
df.head()
df["Type_2"] = df["Type_2"].fillna(df["Type_1"]) # Filling the Nan values of Type 2 to Type1
def wordcloud(df,col):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(background_color="Black",stopwords=stopwords).generate(" ".join([i for i in df[col]]))
    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Good Morning Datascience+")
wordcloud(df,"Type_1")
wordcloud(df,"Type_2")
def title(x,y):
    df.groupby(x).sum().sort_values(by=y,ascending=False)[[y]][:10].plot(kind='bar',figsize=(10,5))
    x= plt.gca().xaxis
    for i in x.get_ticklabels():
        i.set_rotation(75)
title("Type_1","Total")
title("Type_2","Total")
df.groupby(["Type_1","Type_2"]).sum().sort_values(by="Total",ascending=False)[["Total"]][:15].plot(kind='barh',figsize=(10,5))
def bar(x,y,z):
    a=df.groupby([x,y]).count().reset_index()
    a=a[[x,y,z]]
    a=a.pivot(x,y,z)
    a[['Water','Fire','Grass','Dragon','Normal','Rock','Flying','Electric']].plot(color=['b','r','g','#FFA500','brown','#6666ff','#001012','y'],marker='o')
    fig=plt.gcf()
    fig.set_size_inches(12,6)
    plt.show()
bar("Generation","Type_1","Total")
bar("Generation","Type_2","Total")



