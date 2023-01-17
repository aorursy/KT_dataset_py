# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")



df_songs = pd.read_csv("../input/songs.csv")



df_songs_extra = pd.read_csv("../input/song_extra_info.csv")



df_members = pd.read_csv("../input/members.csv",parse_dates=["registration_init_time","expiration_date"])



df_test = pd.read_csv("../input/test.csv")
print("Number of common users in both the datasets : " ,len(set.intersection(set(df_train['msno']), set(df_test['msno']))))
print("Number of Common Songs in both the datasets : ", len(set.intersection(set(df_train['song_id']), set(df_test['song_id']))))



print("No of Unique songs in Training set :", df_train['song_id'].nunique())



print("No of Unique songs in Test set :" ,df_test['song_id'].nunique())
plt.figure(figsize=(12,8))

sns.countplot(df_train['target'])
df_songs.head()
df_songs_extra.head()
df_train =df_train.merge(df_songs,how="left",on="song_id")
df_train =df_train.merge(df_songs_extra,how="left",on="song_id")
df_train.head()
plt.figure(figsize=(12,10))

sns.countplot(df_train['source_system_tab'],hue=df_train['target'])
plt.figure(figsize=(12,10))

g = sns.countplot(df_train['source_type'],hue=df_train['target'])

locs, labels = plt.xticks()

g.set_xticklabels(labels,rotation=45)
df_train.head()
df_train.dropna(subset=["song_length"],inplace=True)



df_train.dropna(subset=["language"],inplace=True)
df_train['source_system_tab'] = df_train['source_system_tab'].astype("category")

df_train['source_type'] = df_train['source_type'].astype("category")
df_train['language'].value_counts()
plt.figure(figsize=(12,10))

sns.countplot(df_train['language'],hue=df_train['target'])
x = df_train['language'].value_counts()
df_len = len(df_train)

for lang_id,count in zip(df_train['language'].value_counts().index,df_train['language'].value_counts()) : 

    

    print(lang_id,":",(100*count / df_len))
df_train = df_train.merge(df_members,how="left",on="msno")
plt.figure(figsize=(14,12))

df_train['bd'].value_counts(sort=False).plot.bar()



plt.xlim([-10,100])
len(df_train.query("bd< 0"))
df_train = df_train.query("bd >= 0")
df_train.head()
len(df_train.query("bd > 100"))
df_train_temp = df_train.query("bd >=5 and bd <80")
df_train_temp['bd'].describe()
plt.figure(figsize=(15,12))

sns.countplot(df_train_temp['bd'])
df_train_temp['age_range'] = pd.cut(df_train_temp['bd'],bins=[5,10,18,30,45,60,80])
plt.figure(figsize=(15,12))

sns.countplot(df_train_temp['age_range'],hue=df_train_temp['target'])
df_train_temp['genre_ids'].value_counts().head()
plt.figure(figsize=(15,12))

sns.boxplot(df_train_temp['age_range'],df_train_temp["song_length"]/60000,hue=df_train_temp['target'],)

plt.ylabel("Song Length in Minutes")

plt.xlabel("Age Groups")

plt.ylim([0,6])
plt.figure(figsize=(14,12))

sns.countplot(df_train_temp['age_range'],hue=df_train_temp["source_type"])

plt.legend(loc="upper right")

plt.figure(figsize=(14,12))

sns.countplot(df_train_temp['age_range'],hue=df_train_temp["source_screen_name"])

plt.legend(loc="upper right")
plt.figure(figsize=(14,12))

sns.countplot(df_train_temp['age_range'],hue=df_train_temp["source_system_tab"])

plt.legend(loc="upper right")
df_train_temp['gender'].value_counts()
plt.figure(figsize=(15,12))

df_train_temp.query("gender =='female'")["genre_ids"].value_counts().head(15).plot.bar()

plt.title("Distribution of Genres across Females ")

plt.xlabel("Genre IDs")

plt.ylabel("Count")
plt.figure(figsize=(15,12))

df_train_temp.query("gender =='male'")["genre_ids"].value_counts().head(15).plot.bar()

plt.title("Distribution of Genres across Males ")

plt.xlabel("Genre IDs")

plt.ylabel("Count")
df_train.drop("composer",axis=1,inplace=True)

df_train_temp.drop("composer",axis=1,inplace=True)
100 * len(df_train.query("bd< 0 or bd >80")) / len(df_train)
df_train = df_train.query("bd> 0 and bd <=80")
df_train.head()
df_test.head()
df_test.drop("composer",axis=1,inplace=True)
df_train.info()
#df_train[''] = pd.cut(df_train['bd'],bins=[5,10,18,30,45,60,80])

df_train['age_range'] = pd.cut(df_train_temp['bd'],bins=[5,10,18,30,45,60,80])
plt.figure(figsize=(14,12))

df_test['bd'].value_counts(sort=False).plot.bar()

plt.xlim([-10,80])