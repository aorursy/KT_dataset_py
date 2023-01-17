!pip install asari
#使いそうなやつとりあえず

import pandas as pd

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime



%matplotlib inline

font = {"family": "Arial"}

matplotlib.rc('font', **font)



#日本語の感情分析用にインポート

#https://github.com/Hironsan/asari）

import asari

from asari.api import Sonar
df = pd.read_csv("../input/shinzo-abe-japanese-prime-minister-twitter-nlp/Shinzo Abe Tweet 20171024 - Tweet.csv")
df.columns
df.head()
#扱いやすいように列名を一部変えます

df.rename(columns= lambda x : str(x).replace(" ","_"),inplace=True)



#Profile_Tweet_1（＝replies）の列を扱いやすいように加工

df.Profile_Tweet_1 = df.Profile_Tweet_1.map(lambda x :str(x).replace(" replies",""))

df.Profile_Tweet_1 = df.Profile_Tweet_1.map(lambda x :str(x).replace(",",""))

df.Profile_Tweet_1 = df.Profile_Tweet_1.astype("int")



#Profile_Tweet_2（＝retweets）の列を扱いやすいように加工

df.Profile_Tweet_2 = df.Profile_Tweet_2.map(lambda x :str(x).replace(" retweets",""))

df.Profile_Tweet_2 = df.Profile_Tweet_2.map(lambda x :str(x).replace(",",""))

df.Profile_Tweet_2 = df.Profile_Tweet_2.astype("int")



#Profile_Tweet_3（＝likes）の列を扱いやすいように加工

df.Profile_Tweet_3 = df.Profile_Tweet_3.map(lambda x :str(x).replace(" likes",""))

df.Profile_Tweet_3 = df.Profile_Tweet_3.map(lambda x :str(x).replace(",",""))

df.Profile_Tweet_3 = df.Profile_Tweet_3.astype("int")



#列名をわかりやすいように変更

df.rename(columns={

   "Profile_Tweet_1":"replies",

   "Profile_Tweet_2":"retweets",

   "Profile_Tweet_3":"likes"

}, inplace = True)



#不要になった列を削除

#"Reply", "Re_Tweet","Like"列は意味が重複するので削除

df.drop(["Reply", "Re_Tweet","Like"], axis=1, inplace=True)
#不要なデータがないか確認

df.Full_Name_Show.unique()
#安倍首相のツイートのみ抽出

df = df[df.Full_Name_Show == "安倍晋三"]

#Tweet_Nav列をdate列に変更

tweet_date = df.Tweet_Nav.str.split(" ",expand=True)

tweet_date.columns = ["month","day","year"]

tweet_date.head()
#確認

for i in tweet_date.columns:

   print(i)

   print(tweet_date[i].unique())
#①year列に2016しかなく、2017の文字列を追加します

tweet_date.year.fillna("2017", inplace=True)

#②2016年と2017年でmonthとdayがずれてる（2016年分がずれてる）ので加工



#まず、一旦2016と2017にそれぞれ分けます

year_2016 = tweet_date[tweet_date.year == "2016"]

year_2017 =  tweet_date[tweet_date.year == "2017"]



#2016年の方をmonthとdayの列名を変更

year_2016.rename(columns={

   "month":"day",

   "day":"month"},inplace = True)



##2017のデータに2016を縦につなげる

tweet_date = pd.concat([year_2017,year_2016],axis=0)



#元のデータフレームにくっつける

df = pd.concat([df,tweet_date],axis=1)



#確認

df.head()
#アルファベットの月名を数字に変更

df["month_int"] = df.month.replace({

   'Oct':10,

   'Sep':9,

   'Feb':2,

   'Dec':12,

   'Nov':11,

   'Jul':7,

   'Jun':6,

   'Apr':4,

   'Mar':3})



#年月日に

df["tweet_date"] = df.day + "/" + df.month + "/" + df.year



#datetimeの型に

df.tweet_date = pd.to_datetime(df.tweet_date)



#いらない列を決しておく

df.drop(["day","month","year","month_int"], axis=1, inplace=True)
#データの型の確認

df.dtypes

#「tweet_date」の列がdatetime64になってることが確認できます
#index列を指定

df = df.set_index("tweet_date")
#マルチインデックスでdfに複数のindex情報をセットすると色々集計しやすい

#年月別ツイート数

df_yqmd = df.set_index([

   df.index.year,

   df.index.quarter,

   df.index.month,

   df.index.day,

   df.index.weekday_name])



#index名を設定

df_yqmd.index.names = ["year","quarter","month","day","day_of_week"]
round(df_yqmd.mean(level=["year","quarter","month"]))
#時系列のmultiindexで年月別のツイート数のカウント(groupby()とsize()を使う）

df_yqmd.groupby(level=["year",'month']).size().plot(kind="bar", colormap='Pastel2', figsize=[10,5])

#何曜日のツイートが多いのだろう

df_yqmd.groupby(level="day_of_week").size().sort_values(ascending=False).plot(kind="bar", colormap='Pastel1', figsize=[10,5])
sns.pairplot(df_yqmd[['replies',"retweets","likes"]]

            # , hue="Day_of_the_week"

             ,markers='+'

            )
sonar = Sonar()



#テキストの感情分析結果の"positive"側の値を取得する

def get_positive_power(text):

    info = sonar.ping(text= text)

    posi_vector = info["classes"][1]["confidence"]

    return posi_vector



#テキストの感情分析結果の"negative"側の値を取得する

def get_negative_power(text):

    info = sonar.ping(text= text)

    nega_vector = info["classes"][0]["confidence"]

    return nega_vector



#テキストのtop_sentiment(どっちの感情が強いか)を取得する

def which_sentiment(text):

    info = sonar.ping(text= text)

    sentiment = info["top_class"]

    return sentiment
#ポジネガ分析してみる

#asariを使って、ツイートの感情値を取得



#まず、プラス

df_yqmd["sentiment_positive"] = df_yqmd.Tweet_Text_Size_Block.map(lambda x : get_positive_power(x))



#ネガティブ

df_yqmd["sentiment_negative"] = df_yqmd.Tweet_Text_Size_Block.map(lambda x : get_negative_power(x))



#ポジネガ

df_yqmd["sentiment"] = df_yqmd.Tweet_Text_Size_Block.map(lambda x : which_sentiment(x))



#sentimentの差し引き

df_yqmd["sentiment_net"] = df_yqmd.sentiment_positive - df_yqmd.sentiment_negative
df_yqmd[["Tweet_Text_Size_Block","sentiment_positive","sentiment_negative","sentiment","sentiment_net"]].head()
#ポジネガの割合

plt.pie(df_yqmd.groupby(by="sentiment").size()/len(df_yqmd), labels=["negative","positive"], autopct="%1.1f%%")

plt.axis('equal')



#ほとんどがポジティブなツイート