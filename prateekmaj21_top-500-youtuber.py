import numpy as np # linear algebra

import pandas as pd # data processing, 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#reading the dataset



df=pd.read_csv("/kaggle/input/youtubers-popularity-dataset/top_500.csv")
#removing the null values



df=df.dropna()
df.head()
df.tail()
len(df)
df.info()
#converting the numbers in format 100,000 to 100000, basically removing the commas



df["Uploads"] = df["Uploads"].str.replace(",","").astype(float)

df["Views"] = df["Views"].str.replace(",","").astype(float)


#cleaning the subscriber column

# 23M now converted to 23 million in numbers

# 145K now converted to 145000 and so on



for i in range(0,495):

    try:

        word=df["Subscriptions"][i]

        l=len(word)

        if (word[l-1]=="M"):

            word=word[:-1]

            word=float(word)

            df["Subscriptions"][i]=word*1000000

        elif (word[l-1]=="K"):

            word=word[:-1]

            word=float(word)

            df["Subscriptions"][i]=word*1000

    except:

        pass

    
#coercing the values into numeric



df['Subscriptions'] = pd.to_numeric(df['Subscriptions'],errors='coerce')

df=df.dropna()
df.head()
df.tail()
df.info()
print("YouTube channels with highest subscribers are = ")

print((df.sort_values("Subscriptions",ascending=False).head(10))['Ch_name'])
print("YouTube channels with lowest subscribers are = ")

print((df.sort_values("Subscriptions",ascending=True).head(10))['Ch_name'])
print("YouTube channels with highest Video Uploads are = ")

print((df.sort_values("Uploads",ascending=False).head(10))['Ch_name'])

print("Interesting thing to note is that channels with most uploads are News Channels.")

print("They have to report daily happenings and have to post multiple videos in a day.")

print('This results in high uploads.')
print("YouTube channels with lowest Video Uploads are = ")

print((df.sort_values("Uploads",ascending=True).head(10))['Ch_name'])
print("YouTube channels with highest Views are = ")

print((df.sort_values("Views",ascending=False).head(10))['Ch_name'])
print("YouTube channels with lowest Views are = ")

print((df.sort_values("Views",ascending=True).head(10))['Ch_name'])
#seeing which YouTubers have 10 million and above subscribers

df1=df[df["Subscriptions"]>=10000000]
print("Number of YouTubers having 10 million and above subscribers are=", len(df1))
#seeing which YouTubers have 50 million and above subscribers

df2=df[df["Subscriptions"]>=50000000]
print("Number of YouTubers having 50 million and above subscribers are=", len(df2))

print("They are=")

df2["Ch_name"]