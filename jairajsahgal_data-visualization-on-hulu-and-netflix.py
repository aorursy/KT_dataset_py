# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv")
del df["Unnamed: 0"]
print(df.head())
#taking movies on HULU

df_hulu=df[df["Hulu"]==1]

df_netflix=df[df["Netflix"]==1]
print("Total Length: ",len(df))

print("Hulu Length: ",len(df_hulu))



print("Total Length: ",len(df))

print("Hulu Length: ",len(df_netflix))

print(df_hulu.head())

print(df_netflix.head())
from collections import Counter

def return_counter(data_frame,column_name,limit):

    print(dict(Counter(data_frame[column_name].values).most_common(limit)))
return_counter(df_hulu,'Language',5)

return_counter(df_hulu,'Genres',5)

return_counter(df_netflix,'Language',5)

return_counter(df_netflix,'Genres',5)
df_d1=df_hulu[df_hulu["Genres"]=="Documentary"]

print(set(df_d1["Title"]))



df_d2=df_netflix[df_netflix["Genres"]=="Documentary"]

print(set(df_d2["Title"]))



print(set(df_d1["Country"]))

print(set(df_d2["Country"]))
print(set(df_d1["Runtime"]))

print(set(df_d2["Runtime"]))
def return_statistics(data_frame,categorical_column,numerical_column):

    mean=[]

    std=[]

    field=[]

    for i in set(list(data_frame[categorical_column].values)):

        new_data=data_frame[data_frame[categorical_column]==i]

        field.append(i)

        mean.append(new_data[numerical_column].mean())

        std.append(new_data[numerical_column].std())

    df=pd.DataFrame({'{}'.format(categorical_column): field,'mean {}'.format(numerical_column): mean,'std in {}'.format(numerical_column): std})

    df.sort_values('mean {}'.format(numerical_column),inplace=True,ascending=False)

    df.dropna(inplace=True)

    return df

stats=return_statistics(df_hulu,"Genres","Runtime")

print(stats.head(15))

stat=return_statistics(df_netflix,"Genres","Runtime")

print(stat.head(15))
import seaborn as sns

def get_boxplot_of_categories(data_frame,categorical_column,numerical_column,limit):

    keys=[]

    for i in dict(Counter(df[categorical_column].values).most_common(limit)):

        keys.append(i)

    print(keys)

    df_new=df[df[categorical_column].isin(keys)]

    sns.set()

    sns.boxplot(x=df_new[categorical_column],y=df_new[numerical_column])

    
get_boxplot_of_categories(df_hulu,"Genres","Runtime",5)



get_boxplot_of_categories(df_netflix,"Genres","Runtime",5)
def get_histogram(data_frame,numerical_column):

    df_new=data_frame

    df_new[numerical_column].hist(bins=100)
get_histogram(df_hulu,"Runtime")
get_histogram(df_netflix,"Runtime")