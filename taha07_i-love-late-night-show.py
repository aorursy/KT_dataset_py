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
import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.io as pio
conan = pd.read_csv("/kaggle/input/late-night-talk-show-youtube-dataset/Conan.csv")

conan.head()
conan.info()
conan.isnull().sum()
conan.drop(["videoDescription","licensedContent"],axis = 1,inplace=True)

conan.head()
profile = pandas_profiling.ProfileReport(conan)

profile
cat = ["videoCategoryId","videoCategoryLabel","durationSec","definition","caption","viewCount","likeCount","dislikeCount","commentCount"]

for value in cat:

    conan[value].fillna(conan[value].mode()[0],inplace=True)
conan.isnull().sum()
sns.set()

plt.figure(figsize=(12,9))

colors=['Red','Blue','Green','Magenta','Orange','Brown','Yellow','Purple','Pink','Cyan']

sns.countplot(x="videoCategoryLabel",data=conan,order = conan["videoCategoryLabel"].value_counts().index[0:10])

plt.xticks(rotation=45)

plt.show()
sns.set()

plt.figure(figsize=(16,9))

sns.catplot(x="videoCategoryLabel",y="likeCount",hue = "videoCategoryLabel",data=conan,aspect=1.6,height=6)

plt.xticks(rotation=45)

plt.show()
sns.set()

plt.figure(figsize=(12,9))

sns.scatterplot(x="dislikeCount",y="commentCount",hue="videoCategoryLabel",data= conan)

fig = px.pie(conan,values = "likeCount",names ="videoCategoryLabel",labels= conan["videoCategoryLabel"],opacity=1)

fig.update_traces(textposition ='inside',textinfo='percent+label')

fig.show()
sns.set()

plt.figure(figsize=(12,9))

sns.scatterplot(x="durationSec",y="viewCount",hue="videoCategoryLabel",data=conan)
sns.set()

plt.figure(figsize=(16,9))

sns.catplot(x="videoCategoryLabel",y="dislikeCount",hue = "videoCategoryLabel",data=conan,aspect=1.6,height=6)

plt.xticks(rotation=45)

plt.show()
com_dis = conan.dislikeCount[conan["videoCategoryLabel"] == "Comedy"].sum()

ent_dis = conan.dislikeCount[conan["videoCategoryLabel"] == "Entertainment"].sum()

game_dis = conan.dislikeCount[conan["videoCategoryLabel"] == "Game"].sum()

mus_dis = conan.dislikeCount[conan["videoCategoryLabel"] == "Music"].sum()
values = [com_dis,ent_dis,mus_dis]

labels= ["Comedy","Entertainment","Music"]

explode=(0.1,0.1,0.9)

color = ['Yellow','Purple','Magenta']

plt.pie(values,explode=explode,radius=1.9,startangle=45,colors=color,labels=labels,autopct="%0.1f%%",shadow = True)

plt.show()
sns.set()

plt.figure(figsize=(25,9))

sns.countplot(x="publishedAtSQL",data= conan,hue= "videoCategoryLabel",order = conan['publishedAtSQL'].value_counts().index[0:20])

plt.xticks(rotation=45)

plt.show()
sum_com = conan[conan["videoCategoryLabel"]=="Comedy"].sum()

sum_music = conan[conan["videoCategoryLabel"]=="Music"].sum()

sum_ent = conan[conan["videoCategoryLabel"]=="Entertainment"].sum()

sum_gam = conan[conan["videoCategoryLabel"]=="Gaming"].sum()

sum_np = conan[conan["videoCategoryLabel"]=="News & Politics"].sum()

sum_te = conan[conan["videoCategoryLabel"]=="Travel & Events"].sum()

sum_pt = conan[conan["videoCategoryLabel"]=="PT1M53S"].sum()

sum_fa = conan[conan["videoCategoryLabel"]=="Film & Animation"].sum()

sum_pb = conan[conan["videoCategoryLabel"]=="People & Blogs"].sum()

sum_edu = conan[conan["videoCategoryLabel"]=="Education"].sum()





prct_com = (sum_com["likeCount"] / sum_com["viewCount"])*100

prct_mus = (sum_music["likeCount"] / sum_music["viewCount"])*100

prct_ent = (sum_ent["likeCount"] / sum_ent["viewCount"])*100

prct_gam = (sum_gam["likeCount"] / sum_gam["viewCount"])*100

prct_te = (sum_te["likeCount"] / sum_te["viewCount"])*100

prct_fa = (sum_fa["likeCount"] / sum_fa["viewCount"])*100

prct_pb = (sum_pb["likeCount"] / sum_pb["viewCount"])*100

prct_edu = (sum_edu["likeCount"] / sum_edu["viewCount"])*100

prct_np =  (sum_np["likeCount"] / sum_np["viewCount"])*100

prct_pt =  (sum_pt["likeCount"] / sum_pt["viewCount"])*100





values=[prct_com,prct_mus,prct_ent,prct_gam,prct_te,prct_fa,prct_pb,prct_edu,prct_np,prct_pt]

labels=['Comedy','Music','Entertainment',"Game","Travel & Events","Film & Animation","People & Blogs","Education","News & Politics","PT1M53S"]

plt.axis('equal')

explode=(0,0.5,0.5,0.2,0.3,0.1,0.2,0.2,0.1,0.1)

colors=['Red','Blue','Green','Magenta','Orange','Brown','Yellow','Purple','Pink','Cyan']

plt.pie(values,radius=2,autopct='%0.1f%%',shadow=True,explode=explode,startangle=45,labels=labels,colors=colors)

plt.show()
cm = conan.groupby("videoCategoryLabel")["commentCount"].sum()

cm.head()
df_cm=pd.DataFrame({"Type": cm.index,"TotalComment":cm.values})

df_cm.nlargest(5,["TotalComment"])
labels=['Comedy','Entertainment','Gaming']

explode=(0,0.1,0.9)

colors=['Pink','Orange','Magenta']

plt.pie(df_cm.nlargest(3,["TotalComment"])["TotalComment"],explode=explode,radius=1.9,colors=colors,startangle=45,labeldistance=1.1,rotatelabels=True,labels=labels,autopct="%0.1f%%",shadow = True)

plt.show()
like_cn = conan[["videoTitle","likeCount","videoCategoryLabel"]]

like = like_cn.nlargest(5,"likeCount")

like
attributes = list(like.videoTitle)

values = list(like.likeCount)

plt.bar(attributes,values,color = ['Magenta','Orange','Brown','Yellow','Purple'])

plt.xticks(rotation = 80)

plt.xlabel("Video Title",fontsize=15)

plt.show()

labels=['Ice Cube, Kevin Hart And Conan Help A Student','James Veitch Is A Terrible Roommate - CONAN on..','Ice Cube, Kevin Hart, And Conan Share A Lyft Car',

       "Disturbed The Sound Of Silence","Jean-Claude Van Damme Recreates His “Kickboxer"]

explode=(0,0.1,0.1,0.1,0.02)

colors=['Pink','Orange','Magenta']

plt.pie(like_cn.nlargest(5,["likeCount"])["likeCount"],explode=explode,radius=1.9,colors=colors,startangle=45,labeldistance=1.1,rotatelabels=True,labels=labels,autopct="%0.1f%%",shadow = True)

plt.show()
comment = conan[["videoTitle","commentCount","videoCategoryLabel"]]

a = comment.nlargest(5,"commentCount")

a
values = list(a.commentCount)

attribute = list(a.videoTitle)

plt.bar(attribute,values,color = ['Magenta','Orange','Brown','Yellow','Purple'])

plt.xticks(rotation = 80)

plt.show()
from wordcloud import WordCloud

plt.figure(figsize=(16,9))

wordcloud = WordCloud(

                          background_color='black',

                          width=1730,

                          height=970

                         ).generate(" ".join(conan.videoTitle))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('videotitle_WC.png')

plt.show() 