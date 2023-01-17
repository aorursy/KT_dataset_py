import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import json

from datetime import timedelta



pd.options.display.float_format = "{:.2f}".format #pandas 출력 포맷팅 참고(https://financedata.github.io/posts/pandas-display-format.html)



%matplotlib inline
data = pd.read_csv("/kaggle/input/youtube-new/KRvideos.csv", engine = "python")
data.info()
df = data[:]



for col in ["thumbnail_link", "comments_disabled", "ratings_disabled", "video_error_or_removed"]:#해당 분석에서 사용하지 않아 미리 삭제

    del df[col] 
df["video_id"].duplicated().value_counts()
df = df.drop_duplicates(["video_id"])[:]
df["video_id"].duplicated().value_counts()
id_to_category = {}



with open("/kaggle/input/youtube-new/KR_category_id.json","r") as f:

    id_data = json.load(f)

    for category in id_data["items"]:

        id_to_category[category["id"]] = category["snippet"]["title"]



# id_to_category



df["category_id"] = df["category_id"].astype(str)

df.insert(4, "category", df["category_id"].map(id_to_category))
df.isnull().sum()
df["category_id"].loc[df["category"].isnull() == True].value_counts()
df["category"].loc[df["category"].isnull() == True] = "Nonprofits & Activism"
data["trending_date"] = pd.to_datetime(data["trending_date"], format = "%y.%d.%m")



df["trending_date"] = pd.to_datetime(df["trending_date"], format = "%y.%d.%m")



df["publish_time"] = pd.to_datetime(df["publish_time"])

df["publish_time"] = df["publish_time"].apply(lambda x : x.date())

df["publish_time"] = pd.to_datetime(df["publish_time"])
df["tags"][df["video_id"] == "C-bLqIftDY4"].values
df["tag_count"] = df["tags"].apply(lambda x : len(x.split("|")) if x != "[none]" else 0)
df["likes/views"] = df["likes"] / df["views"]

df["dislikes/views"] = df["dislikes"] / df["views"]

df["comment_count/views"] = df["comment_count"] / df["views"]

df["dislikes/likes"] = df["dislikes"] / df["likes"]

df["dislikes/likes"].loc[df["dislikes/likes"] == np.inf] = 0
df.isnull().sum()
df["dislikes/likes"].loc[df["dislikes/likes"].isnull() == True] = 0
df["title_length"] = df["title"].apply(lambda x : len(str(x)) if pd.isnull(x) == False  else 0 )
df["description_length"] = df["description"].apply(lambda x : len(str(x)) if pd.isnull(x) == False  else 0 )
df["treTime-pubTime"] = df["trending_date"] - df["publish_time"] + timedelta(days = 1)
df.info()
print("분석에 사용한 데이터는 {}개의 변수를 가진 인기동영상 데이터 {}개입니다.".format(len(df.columns), format(len(df),",")))
df.describe()
plt.figure(figsize = (20, 20))

for i in range(len(df.describe().columns) - 1 ):

    plt.subplot(4, 4, i + 1)

    plt.title(df.describe().columns[i])

    plt.hist(df[df.describe().columns[i]],bins = 10)

    plt.grid()
def summary_numeric(col):

    print(col)

    print("평균 {:>15}".format(format(df[col].mean(), ",.2f")))

    print("최소 {:>15}".format(format(df[col].min(), ",.2f")))

    print("중앙 {:>15}".format(format(df[col].median(), ",.2f")))

    print("최대 {:>15}".format(format(df[col].max(), ",.2f")))
summary_numeric("views")
df[["video_id", "title", "channel_title", "category", "views", "likes", "dislikes", "comment_count", "publish_time", "trending_date"]].sort_values(by = ["views"], ascending = [False]).head(3)
print("전체 데이터 {}건".format(len(df)) )

print("1000만이상 {}건 전체의 {:.2f}%".format(len(df.loc[df["views"] >= 10000000]),len(df.loc[df["views"] >= 10000000]) / len(df) * 100))

print("100만이상 {}건 전체의 {:.2f}%".format(len(df.loc[df["views"] >= 1000000]),len(df.loc[df["views"] >= 1000000]) / len(df) * 100))

print("100만이하 {}건 전체의 {:.2f}%".format(len(df.loc[df["views"] < 1000000]),len(df.loc[df["views"] < 1000000]) / len(df) * 100))
print("조회수 100만 이하 데이터의 평균: {:.2f}".format(df["views"].loc[df["views"] <= 1000000].mean()))

plt.hist(df["views"].loc[df["views"] <= 1000000]);
df.loc[df["views"] <= 3000,["video_id", "title", "publish_time", "trending_date", "category", "views", "likes", "comment_count", "tag_count"]].sort_values(by = "views")
df[["video_id", "title", "channel_title", "category", "views", "likes", "dislikes", "comment_count", "tag_count", "description_length"]].sort_values(by = ["likes"], ascending = [False]).head(3)
df[["video_id", "title", "channel_title", "category", "views", "likes", "dislikes", "comment_count", "tag_count", "description_length"]].sort_values(by = ["dislikes"], ascending = [False]).head(3)
df[["video_id", "title", "channel_title", "category","views","likes", "dislikes", "comment_count", "tag_count", "description_length"]].sort_values(by = ["comment_count"], ascending = [False]).head(3)
plt.hist(df["tag_count"],bins = 10);
df[["video_id", "title", "channel_title", "category", "views", "likes", "dislikes", "comment_count", "tag_count","description_length"]].sort_values(by = ["tag_count"], ascending = [False]).head(10)
df[["likes/views", "dislikes/views", "comment_count/views", "dislikes/likes"]].describe()
df[["video_id", "title", "channel_title", "category", "views", "likes/views", "dislikes/views", "comment_count/views", "dislikes/likes", "trending_date"]].sort_values(by = ["likes/views"], ascending = [False]).head(5)
df[["video_id", "title", "channel_title", "category", "views", "likes/views", "dislikes/views", "comment_count/views", "dislikes/likes", "trending_date"]].sort_values(by = ["dislikes/views"], ascending = [False]).head(5)
df[["video_id", "title", "channel_title", "category", "views", "likes", "dislikes", "dislikes/likes", "trending_date"]].sort_values(by = ["dislikes/likes"], ascending = [False]).head(5)
plt.hist(df["title_length"],bins = 10)

plt.axvline(x = 42, color = "r", linestyle = "-", linewidth = 1)
df.loc[df["title_length"] <= 2]
plt.hist(df["description_length"],bins = 10);
df[["video_id", "title", "channel_title", "category", "views", "likes", "dislikes", "comment_count", "tag_count", "description_length"]].sort_values(by = ["description_length"], ascending = [False]).head(10)
plt.figure(figsize = (15,10));

data["trending_date"].value_counts().plot(label = "Number of Trending videos per day, including duplicates")

df["trending_date"].value_counts().plot(label = "Number of Trending videos per day, excluding duplicates")



plt.legend(prop = {"size":17});
print("중복영상을 포함한 하루 인기동영상의 개수")

print("평균 : {:.2f}개".format(data["trending_date"].value_counts().mean()))

print("최대 : {:.2f}개".format(data["trending_date"].value_counts().max()))

print("최소 : {:.2f}개".format(data["trending_date"].value_counts().min()))



print("중복영상을 제외한 하루 인기동영상의 개수")

print("평균 : {:.2f}개".format(df["trending_date"].value_counts().mean()))

print("최대 : {:.2f}개".format(df["trending_date"].value_counts().max()))

print("최소 : {:.2f}개".format(df["trending_date"].value_counts().min()))
df["treTime-pubTime"].describe()
pd.cut(df["treTime-pubTime"],[timedelta(days = 0),timedelta(days = 1),timedelta(days = 2),timedelta(days = 3),timedelta(days = 2335)]).value_counts(sort = False,normalize = True)
df.loc[df["treTime-pubTime"] > timedelta(days = 1000)].sort_values(by = "treTime-pubTime", ascending = False) 
print("전체 데이터 {}개\ncategory 개수 {}개\n".format(len(df),len(df["category"].unique())))

for i in range(len(df["category"].value_counts())):

    print("{:23} {:4} {:.2f}".format(df["category"].value_counts().index[i],

                            df["category"].value_counts().values[i],

                            df["category"].value_counts(normalize = True).values[i]))
for cate in df["category"].value_counts().index[0:3]:

    print(cate, df["treTime-pubTime"].loc[df["category"] == cate].mean())
print("전체 데이터 개수 {} 포함된 채널의 수 {}".format(len(df), len(df["channel_title"].unique())))   
df_pivot=pd.pivot_table(df, index = ["channel_title"], 

                    values = ["views"],

                    aggfunc = ["count", "mean"])
df_pivot["count"]["views"].value_counts().plot(kind = "pie",title = "Channels by Number of Trending Videos",label = " ",figsize = (6, 6));
df["category"].loc[df["channel_title"].isin(df_pivot[df_pivot["count"]["views"] == 1].index)].value_counts()
df["category"].loc[df["channel_title"].isin(df_pivot[(df_pivot["count"]["views"] >= 2) & (df_pivot["count"]["views"] < 4)].index)].value_counts()
df["category"].loc[df["channel_title"].isin(df_pivot[df_pivot["count"]["views"] >= 4].index)].value_counts()
plt.figure(figsize = (24, 24))



for i in range(1, len(df.describe().columns) - 1 ):

    plt.subplot(4, 4, i)

    plt.title("relationship between views and {}".format(df.describe().columns[i]))

    plt.xlabel("views")

    plt.ylabel(df.describe().columns[i])

    plt.scatter(x = df["views"], y = df[df.describe().columns[i]])

    plt.grid()
plt.figure(figsize = (24, 24))



for i in range(1, len(df.describe().columns) - 1 ):

    plt.subplot(4, 4, i)

    plt.title("relationship between views and {}".format(df.describe().columns[i]))

    plt.xlabel("views")

    plt.ylabel(df.describe().columns[i])

    plt.scatter(x = df["views"].loc[df["views"] <= 1000000], y = df[df.describe().columns[i]].loc[df["views"] <= 1000000])

    plt.grid()
plt.figure(figsize = (15,15))

sns.heatmap(data = df.loc[df["views"] <= 1000000].corr(method="pearson"), annot = True, fmt = ".2f", linewidths = .5, cmap = "Blues");