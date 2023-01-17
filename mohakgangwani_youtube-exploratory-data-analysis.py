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
#Importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
IN = pd.read_csv("../input/youtube-new/INvideos.csv")
IN
IN.shape
IN.info()
#Converting trending_date, publish_time to Datetime format
IN["trending_date"] = pd.to_datetime(IN["trending_date"],format="%y.%d.%m")
IN["publish_time"] = pd.to_datetime(IN["publish_time"])
IN.describe().apply(lambda s: s.apply(lambda x: format(x, "f")))
IN.isnull().sum()
IN[IN["description"].isnull()]
IN.loc[IN["description"].isnull(),"description"] = " "
IN["tags"] = IN["tags"].str.replace('|',",").str.replace('"',"")
IN_Category = pd.read_json("../input/youtube-new/IN_category_id.json")
ID_G = []
keys = []
items = []
for i in list(IN_Category["items"]):
    ID = i["id"]
    genre = i["snippet"]["title"]
    ID_G.append((ID,genre))
ID_G.append(('29','Events and Motivational'))
IN["video_type"]= [dict(ID_G)[x] for x in list(map(str,IN["category_id"]))]
#Here, we could've used video_id. But we see that video_id of more that 500 records is missing.
#So, it would be wise to use title instead.
IN["trending_since(in days)"] = [len(IN[IN["title"]==i]) for i in IN["title"]]
#Records contain everyday updates of each trending video. We will drop all the duplicate records and keep the last one since, it contains total number of likes, dislikes, views,and comm
IN = IN.drop_duplicates(subset="title",keep="last")
IN
IN.reset_index(drop=True,inplace=True)
IN["text"] = IN["title"]+" "+IN["tags"]+" "+IN["description"]
IN["trending_month"] = pd.DatetimeIndex(IN['trending_date']).month
IN.columns
cols = ['trending_date','trending_since(in days)', 'trending_month', 'title',
        'channel_title', 'video_type', 'publish_time', 'tags', 'views',
        'likes', 'dislikes', 'comment_count', 'thumbnail_link',
        'comments_disabled', 'ratings_disabled', 'video_error_or_removed',
        'description', 'text']
IN = IN[cols]
df = IN.copy()
df
df.describe()
f, axes = plt.subplots(2,2,figsize=(12,9))
sns.set_style("whitegrid")
f.suptitle('Histograms of Views,Likes,Dislikes,Comment Count.', fontsize=30)

sns.distplot(df["views"], ax=axes[0, 0])
axes[0,0].set_title('Views',size=20)

sns.distplot(df["likes"], ax=axes[0, 1])
axes[0,1].set_title('Likes',size=20)

sns.distplot(df["dislikes"], ax=axes[1, 0])
axes[1,1].set_title('Dislikes',size=20)

sns.distplot(df["comment_count"], ax=axes[1, 1])
axes[1,0].set_title('Comment Count',size=20)

plt.show()
#Creating a dictionary to get a DataFrame of the Quantiles
d = dict([(f"{i}",np.quantile([df[f"{i}"]],[0,0.1,0.25,0.5,0.75,0.9,0.95,1])) for i in ["views","likes","dislikes","comment_count"]])
pd.DataFrame(d , index = [0,0.1,0.25,0.5,0.75,0.9,0.95,1])
plt.figure(figsize=(32,9))
sns.barplot(df["video_type"].unique(),df["video_type"].value_counts(sort=False))
plt.show()
plt.figure(figsize = (10,6))

sns.countplot(x='comments_disabled', data=df)
plt.title("Comments Disabled", fontsize=20)

plt.show()
plt.figure(figsize = (10,6))

sns.countplot(x='ratings_disabled', data=df)
plt.title("Ratings Disabled", fontsize=20)

plt.show()
plt.figure(figsize = (10,6))

sns.countplot(x='video_error_or_removed', data=df)
plt.title("Video Error or Removed", fontsize=20)

plt.show()
sns.pairplot(df.loc[:,["views","likes","dislikes","comment_count"]],height=3).fig.suptitle("Pair Plot",y = 1.05, fontsize=50)
plt.show()
plt.figure(figsize = (16,9))
sns.heatmap(df.corr(), annot=True, cmap="viridis")
plt.show()
#List of all unique channel title
cats = df["video_type"].unique()
#Sum of Like received on each video of various channels
vals = np.array([len(df[df["video_type"]==i]) for i in df["video_type"].unique()])
#Soring by descending order and storing their index no.
sort = cats[np.argsort(vals)[::-1]]
#Plotting the Bar Graph
plt.figure(figsize=(16,9))
sns.set(style="whitegrid")
sns.barplot(vals,cats,orient='h',palette="cool",order=sort)
plt.xlabel("Number of Trending Videos on YouTube")
plt.ylabel("Category of the Video")
plt.title("WHAT CATEGORY OF VIDEOS ARE MOST TRENDING IN INDIA ?")
plt.show()
#List of all unique channel title
cats = df["video_type"].unique()
#Sum of Like received on each video of various channels
vals = np.array([df.groupby("video_type").get_group(i)["views"].sum() for i in cats])
#Soring by descending order and storing their index no.
sort = np.argsort(vals)[::-1]
#Ordering the list of cats and vals according to the indexes stored and choosing top 25 values
cats=cats[sort][:25]
vals=vals[sort][:25]
#Plotting the Bar Graph
plt.figure(figsize=(16,9))
sns.set(style="whitegrid")
sns.barplot(vals,cats,orient='h',palette="cool")
plt.xlabel("Total number of views on all videos of each Category")
plt.ylabel("Category Name")
plt.title("WHICH CATEGORY DO CANADIANS VIEW THE MOST ?")
plt.show()
#List of all unique channel title
cats = df["video_type"].unique()
#Sum of Like received on each video of various channels
vals = np.array([(df.groupby("video_type").get_group(i)["likes"]-df.groupby("video_type").get_group(i)["dislikes"]).sum() for i in cats])
#Soring by descending order and storing their index no.
sort = np.argsort(vals)[::-1]
#Ordering the list of cats and vals according to the indexes stored and choosing top 25 values
cats=cats[sort][:25]
vals=vals[sort][:25]
#Plotting the Bar Graph
plt.figure(figsize=(16,9))
sns.set(style="whitegrid")
sns.barplot(vals,cats,orient='h',palette="cool")
plt.xlabel("Total number of likes on all videos of each category")
plt.ylabel("Category Name")
plt.title("WHICH CATEGORY DO INDIANS LIKE THE MOST ?")
plt.show()
#List of all unique channel title
cats = df["video_type"].unique()
#Sum of Like received on each video of various channels
vals = np.array([df.groupby("video_type").get_group(i)["comment_count"].sum() for i in cats])
#Soring by descending order and storing their index no.
sort = np.argsort(vals)[::-1]
#Ordering the list of cats and vals according to the indexes stored and choosing top 25 values
cats=cats[sort][:25]
vals=vals[sort][:25]
#Plotting the Bar Graph
plt.figure(figsize=(16,9))
sns.set(style="whitegrid")
sns.barplot(vals,cats,orient='h',palette="cool")
plt.xlabel("Total number of comments on all videos of each Category")
plt.ylabel("Category Name")
plt.title("ON WHICH CATEGORY DO INDIANS COMMENT THE MOST ?")
plt.show()
#List of all unique channel title
cats = df["video_type"].unique()
#Sum of Like received on each video of various channels
vals = np.array([(df.groupby("video_type").get_group(i)["dislikes"]-df.groupby("video_type").get_group(i)["likes"]).sum() for i in cats])
#Soring by descending order and storing their index no.
sort = np.argsort(vals)[::-1]
#Ordering the list of cats and vals according to the indexes stored and choosing top 25 values
cats=cats[sort][:25]
vals=vals[sort][:25]
#Plotting the Bar Graph
plt.figure(figsize=(16,9))
sns.set(style="whitegrid")
sns.barplot(vals,cats,orient='h',palette="cool")
plt.xlabel("Total number of dislikes on all videos of each Category")
plt.ylabel("Category Name")
plt.title("WHICH CATEGORY DO INDIAND DISLIKE THE MOST ?")
plt.show()
#List of all unique channel title
cats = df["channel_title"].unique()
#Sum of Like received on each video of various channels
vals = np.array([len(df[df["channel_title"]==i]) for i in df["channel_title"].unique()])
#Soring by descending order and storing their index no.
sort = np.argsort(vals)[::-1]
#Ordering the list of cats and vals according to the indexes stored and choosing top 25 values
cats=cats[sort][:25]
vals=vals[sort][:25]
#Plotting the Bar Graph
plt.figure(figsize=(16,9))
sns.set(style="whitegrid")
sns.barplot(vals,cats,orient='h',palette="cool")
plt.xlabel("Number of Trending Videos on YouTube")
plt.ylabel("Channel Name")
plt.title("WHICH CHANNEL IS MOST TRENDING ON YOUTUBE IN INDIA ?")
plt.show()
#List of all unique channel title
cats = df["channel_title"].unique()
#Sum of Like received on each video of various channels
vals = np.array([df.groupby("channel_title").get_group(i)["views"].sum() for i in cats])
#Soring by descending order and storing their index no.
sort = np.argsort(vals)[::-1]
#Ordering the list of cats and vals according to the indexes stored and choosing top 25 values
cats=cats[sort][:25]
vals=vals[sort][:25]
#Plotting the Bar Graph
plt.figure(figsize=(16,9))
sns.set(style="whitegrid")
sns.barplot(vals,cats,orient='h',palette="cool")
plt.xlabel("Total number of views on all videos of each Channel")
plt.ylabel("Channel Name")
plt.title("WHICH CHANNEL DO INDIANS VIEW THE MOST ?")
plt.show()
#List of all unique channel title
cats = df["channel_title"].unique()
#Sum of Like received on each video of various channels
vals = np.array([(df.groupby("channel_title").get_group(i)["likes"]-df.groupby("channel_title").get_group(i)["dislikes"]).sum() for i in cats])
#Soring by descending order and storing their index no.
sort = np.argsort(vals)[::-1]
#Ordering the list of cats and vals according to the indexes stored and choosing top 25 values
cats=cats[sort][:25]
vals=vals[sort][:25]
#Plotting the Bar Graph
plt.figure(figsize=(16,9))
sns.set(style="whitegrid")
sns.barplot(vals,cats,orient='h',palette="cool")
plt.xlabel("Total number of likes on all videos of each Channel")
plt.ylabel("Channel Name")
plt.title("WHICH CHANNEL DO INDIANS LIKE THE MOST ?")
plt.show()
#List of all unique channel title
cats = df["channel_title"].unique()
#Sum of Like received on each video of various channels
vals = np.array([df.groupby("channel_title").get_group(i)["comment_count"].sum() for i in cats])
#Soring by descending order and storing their index no.
sort = np.argsort(vals)[::-1]
#Ordering the list of cats and vals according to the indexes stored and choosing top 25 values
cats=cats[sort][:25]
vals=vals[sort][:25]
#Plotting the Bar Graph
plt.figure(figsize=(16,9))
sns.set(style="whitegrid")
sns.barplot(vals,cats,orient='h',palette="cool")
plt.xlabel("Total number of comments on all videos of each Channel")
plt.ylabel("Channel Name")
plt.title("ON WHICH CHANNEL DO INDIANS COMMENT THE MOST ?")
plt.show()
#List of all unique channel title
cats = df["channel_title"].unique()
#Sum of Like received on each video of various channels
vals = np.array([(df.groupby("channel_title").get_group(i)["dislikes"]-df.groupby("channel_title").get_group(i)["likes"]).sum() for i in cats])
#Soring by descending order and storing their index no.
sort = np.argsort(vals)[::-1]
#Ordering the list of cats and vals according to the indexes stored and choosing top 25 values
cats=cats[sort][:25]
vals=vals[sort][:25]
#Plotting the Bar Graph
plt.figure(figsize=(16,9))
sns.set(style="whitegrid")
sns.barplot(vals,cats,orient='h',palette="cool")
plt.xlabel("Total number of dislikes on all videos of each Channel")
plt.ylabel("Channel Name")
plt.title("WHICH CHANNEL DO INDIANS DISLIKE THE MOST ?")
plt.show()
from wordcloud import WordCloud, STOPWORDS
def PlotWordCloud(data,category):
    
    

    text_words = '' 
    stopwords = set(STOPWORDS)

    #Iterate through the csv file 
    for val in data[data["video_type"]==category].sort_values(by="views").reset_index(drop=True).loc[:100,"text"]:
        
        #Typecaste each val to string
        val = str(val)
        
        #Split the value 
        tokens = val.split()
        
        #Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower()
        
        text_words += " ".join(tokens)+" "
    
    text_words = text_words.replace("bit"," ").replace("http"," ").replace("https"," ").replace("com"," ").replace("youtube"," ").replace("gmail"," ").replace("ly"," ").replace("www"," ").replace("youtu"," ").replace("be"," ").replace("goo"," ")
    
    wordcloud1 = WordCloud(width = 1600, height = 900, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(text_words)

    
    
    
    
    text_words = '' 
    stopwords = set(STOPWORDS)

    #Iterate through the csv file 
    for val in data[data["video_type"]==category].sort_values(by="likes").reset_index(drop=True).loc[:100,"text"]:
        
        #Typecaste each val to string
        val = str(val)
        
        #Split the value 
        tokens = val.split()
        
        #Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower()
        
        text_words += " ".join(tokens)+" "
    
    text_words = text_words.replace("bit"," ").replace("http"," ").replace("https"," ").replace("com"," ").replace("youtube"," ").replace("gmail"," ").replace("ly"," ").replace("www"," ").replace("youtu"," ").replace("be"," ").replace("goo"," ")
    
    wordcloud2 = WordCloud(width = 1600, height = 900, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(text_words)

    
    
    
    
    
    text_words = '' 
    stopwords = set(STOPWORDS)

    #Iterate through the csv file 
    for val in data[data["video_type"]==category].sort_values(by="comment_count").reset_index(drop=True).loc[:100,"text"]:
        
        #Typecaste each val to string
        val = str(val)
        
        #Split the value 
        tokens = val.split()
        
        #Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower()
        
        text_words += " ".join(tokens)+" "
    
    text_words = text_words.replace("bit"," ").replace("http"," ").replace("https"," ").replace("com"," ").replace("youtube"," ").replace("gmail"," ").replace("ly"," ").replace("www"," ").replace("youtu"," ").replace("be"," ").replace("goo"," ")
    
    wordcloud3 = WordCloud(width = 1600, height = 900, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(text_words)
    
    #Plot the WordCloud images
    f, axarr = plt.subplots(3,1,figsize=(16,27))
    f.suptitle(f'{category}', fontsize=50)
    axarr[0].imshow(wordcloud1)
    axarr[0].axis("off")
    axarr[0].set_title('On the basis of Views',size=30)
    axarr[1].imshow(wordcloud2)
    axarr[1].axis("off")
    axarr[1].set_title('On the basis of Likes',size=30)
    axarr[2].imshow(wordcloud3)
    axarr[2].axis("off")
    axarr[2].set_title('On the basis of Comment Count',size=30)
df["video_type"].value_counts()
PlotWordCloud(df,"Entertainment")
PlotWordCloud(df,"News & Politics")
PlotWordCloud(df,"People & Blogs")
PlotWordCloud(df,"Music")
PlotWordCloud(df,"Comedy")
#Creating a new dataframe that contains the log values of Views, Likes, Comment Count, Dislikes
log = pd.DataFrame({"log views":np.log(df['views']+1),"log likes":np.log(df['likes']+1),"log dislikes":np.log(df['dislikes']+1),"log comments":np.log(df['comment_count']+1),"video_type":df["video_type"]})
log
sns.set_style("whitegrid")
sns.FacetGrid(log,height=7,aspect=32/9).map(sns.boxplot,x=log["video_type"],y=log["log views"],order=log["video_type"].unique(),palette="Set1").add_legend()
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(log,height=7,aspect=32/9).map(sns.boxplot,x=log["video_type"],y=log["log likes"],order=log["video_type"].unique(),palette="Set1").add_legend()
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(log,height=7,aspect=32/9).map(sns.boxplot,x=log["video_type"],y=log["log comments"],order=log["video_type"].unique(),palette="Set1").add_legend()
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(log,height=7,aspect=32/9).map(sns.boxplot,x=log["video_type"],y=log["log dislikes"],order=log["video_type"].unique(),palette="Set1").add_legend()
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(df,height=7,aspect=32/9).map(sns.boxplot,x=df["video_type"],y=df["trending_since(in days)"],order=log["video_type"].unique(),palette="Set1").add_legend()
plt.show()
sns.barplot(x = np.sort(df["trending_month"].unique()),y = df["trending_month"].value_counts(sort=False),palette="cool")
plt.show()
for i in df["video_type"].unique():
    plt.figure(figsize=(16,9))
    sns.barplot(x=np.sort(df.groupby("video_type").get_group(f"{i}")["trending_month"].unique()),y=df.groupby("video_type").get_group(f"{i}")["trending_month"].value_counts(sort=False),palette="cool")
    plt.xlabel("Trending Month")
    plt.ylabel(f"No. of Trending Videos of {i} category")
    plt.title(f"{i}")
    plt.show()
