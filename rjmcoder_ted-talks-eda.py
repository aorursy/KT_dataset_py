!pip install seaborn --upgrade



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



print("Seaborn version used is {}".format(sns.__version__))

print("Pandas version used is {}".format(pd.__version__))

ted_data = pd.read_csv("/kaggle/input/ted-talks/ted_main.csv")

ted_data.head()
display(ted_data.shape)

display(ted_data.describe())

display(ted_data.info())
ted_data.isnull().sum()
def date_convert(x):

    return pd.to_datetime(x,unit='s')



ted_data['film_date'] = ted_data['film_date'].apply(date_convert)

ted_data['published_date'] = ted_data['published_date'].apply(date_convert)

ted_data['duration'] = ted_data['duration'].apply(lambda x: round(x/60, 2))

display(ted_data.head())
print("Average duration = {}".format(round(ted_data["duration"].mean(), 2)))



print("Median duration = {}".format(round(ted_data["duration"].median(), 2)))



display(ted_data[["main_speaker", "title", "event", "duration"]].sort_values("duration", ascending=False).head(10))



plt.figure(figsize=(10,5))

ax = sns.barplot(x="duration", y="main_speaker", data=ted_data.sort_values('duration', ascending=False)[:10])

ax.set_title("Top duration", pad=10, fontdict={'fontsize': 20})

plt.show()
ted_data[["main_speaker", "title", "event", "duration"]].sort_values("duration").head(10)
print("Average views = {}".format(round(ted_data["views"].mean(), 2)))



print("Median views = {}".format(round(ted_data["views"].median(), 2)))



display(ted_data[["main_speaker", "title", "event", "views"]].sort_values('views', ascending=False).head(10))



plt.figure(figsize=(10,5))

ax = sns.barplot(x="views", y="main_speaker", data=ted_data.sort_values('views', ascending=False)[:10])

ax.set_title("Top Viewed", pad=10, fontdict={'fontsize': 20})

plt.show()
print("Average comments = {}".format(round(ted_data["comments"].mean(), 2)))



print("Median comments = {}".format(round(ted_data["comments"].median(), 2)))



plt.figure(figsize=(10,5))

display(ted_data[["main_speaker", "title", "event", "comments"]].sort_values('comments', ascending=False).head(10))

ax = sns.barplot(x="comments", y="main_speaker", data=ted_data.sort_values('comments', ascending=False)[:10])

ax.set_title("Top Comments", pad=10, fontdict={'fontsize': 20})

plt.show()
sns.pairplot(data=ted_data, vars=["views", "comments", "duration"])

display(ted_data[["views", "comments", "duration"]].corr())
top_viewed = ted_data[["name", "title", "views"]].sort_values("views", ascending=False).head(10)

top_commented = ted_data[["name", "comments"]].sort_values("comments", ascending=False).head(10)



top_viewed.merge(top_commented, on="name")
print("Total no. of speakers = {}".format(ted_data["speaker_occupation"].value_counts().sum()))

display(ted_data["speaker_occupation"].value_counts(normalize=True).head())



df = pd.DataFrame(data=ted_data["speaker_occupation"].value_counts().head(10))



df.reset_index(inplace=True)

df.columns = ["speaker_occupation", "count"]



plt.figure(figsize=(15,5))

ax = sns.barplot(data=df, x="speaker_occupation", y="count")

ax.set_title("Top speaker occupation", pad=10, fontdict={'fontsize': 20})

plt.show()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ted_data['month'] = ted_data['film_date'].apply(lambda x: months[x.month - 1])

ted_data['year'] = ted_data['film_date'].apply(lambda x: x.year)

talk_months = ted_data['month'].value_counts().reset_index()



talk_months.columns = ["month", "no_of_talks"]



plt.figure(figsize=(15,5))

ax = sns.barplot(x="month", y="no_of_talks", data=talk_months, order=months)

ax.set_title("Number of talks over months", pad=10, fontdict={'fontsize': 20})

plt.show()
talk_years = ted_data['year'].value_counts().reset_index()



talk_years.columns = ["year", "no_of_talks"]



plt.figure(figsize=(18,5))

ax = sns.barplot(x="year", y="no_of_talks", data=talk_years)

ax.set_title("Number of talks over years", pad=10, fontdict={'fontsize': 20})

plt.show()
# print first few values in the ratings column to understand its structure.

for i in ted_data['ratings'][0:2]:

    print("Value: {}".format(i))

    print("Type: {}".format(type(i)))

    print("\n")

import ast

ted_data["ratings"] = ted_data["ratings"].apply(lambda x: ast.literal_eval(x))
for i in ted_data['ratings'][0:2]:

    print("Value: {}".format(i))

    print("Type: {}".format(type(i)))

    print("\n")
ratings_list = []

for x in ted_data["ratings"]:

    d = (pd.json_normalize(x)

    .drop(columns="id")

    .set_index(keys="name")

     .T)

    ratings_list.append(d)



ratings_df = pd.concat(ratings_list)

ratings_df.reset_index(drop=True, inplace=True)

ratings_df.head()
speaker_ratings_df = pd.concat([ted_data[["main_speaker", "title"]], ratings_df], axis=1)
display(speaker_ratings_df[["main_speaker", "title", "Funny"]].sort_values("Funny", ascending=False).head(10))



plt.figure(figsize=(15,5))

ax = sns.barplot(data=speaker_ratings_df.sort_values("Funny", ascending=False).head(10), 

            y="main_speaker", x="Funny", ci=None) # ci=None is needed to remove the error bars

ax.set_title("Top Funny talks", pad=10, fontdict={'fontsize': 20})

plt.show()
display(speaker_ratings_df[["main_speaker", "title", "Confusing"]].sort_values("Confusing", ascending=False).head(10))



plt.figure(figsize=(15,5))

ax = sns.barplot(data=speaker_ratings_df.sort_values("Confusing", ascending=False).head(10), 

            y="main_speaker", x="Confusing", ci=None) # ci=None is needed to remove the error bars

ax.set_title("Top Confusing talks", pad=10, fontdict={'fontsize': 20})

plt.show()
display(speaker_ratings_df[["main_speaker", "title", "Inspiring"]].sort_values("Inspiring", ascending=False).head(10))



plt.figure(figsize=(15,5))

ax = sns.barplot(data=speaker_ratings_df.sort_values("Inspiring", ascending=False).head(10), 

            y="main_speaker", x="Inspiring", ci=None) # ci=None is needed to remove the error bars

ax.set_title("Top Inspiring talks", pad=10, fontdict={'fontsize': 20})

plt.show()
# print first few values in the tags column to understand its structure.

for i in ted_data['tags'][0:2]:

    print("Value: {}".format(i))

    print("Type: {}".format(type(i)))

    print("\n")
ted_data["tags"] = ted_data["tags"].apply(lambda x: ast.literal_eval(x))
for i in ted_data['tags'][0:2]:

    print("Value: {}".format(i))

    print("Type: {}".format(type(i)))

    print("\n")
tags = ted_data["tags"].explode().value_counts().head(10)

display(ted_data["tags"].explode().value_counts(normalize=True).head(10))

display(ted_data["tags"].explode().value_counts(normalize=True).head(10).sum())

plt.figure(figsize=(15,5))

ax = sns.barplot(x=tags.index, y=tags.values)

ax.set_title("Top tags", pad=10, fontdict={'fontsize': 20})

plt.show()