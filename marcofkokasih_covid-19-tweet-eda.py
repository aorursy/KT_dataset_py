import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import figure
df = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')

df.head()
df.info()
df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d %H:%M:%S')

df['user_created'] = pd.to_datetime(df['user_created'], format='%Y-%m-%d %H:%M:%S')
print("So right now, we currently know that:\n")

print("Number of tweets: {}\n".format(df.shape[0]))

print("Number of users: {}\n".format(df.user_name.nunique()))

print("Users with more than 100K followers: {}\n".format(df[df['user_followers']>100000].user_name.nunique()))

print("Number of verified users: {}\n".format(df[df['user_verified']==True].user_name.nunique()))
def plot_count(x,df,title,xlabel,ylabel):

    figure(figsize=(20, 6))

    sns.set_style("whitegrid")

    

    total = float(len(df))

    ax = sns.countplot(df[x],order=df[x].value_counts().index[:10])

    for i in ax.patches:

        height = i.get_height()

        ax.text(i.get_x()+i.get_width()/2.,

               height + 3,

               '{:1.2f}%'.format(100*height/total),

               ha="center")

    

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    plt.show()  
plot_count('source',df, "Top 10 Source of Tweet", "Source of Tweets", "Number of Tweets")
plot_count('user_location',df, "Top 10 User Locations", "Tweets Locations", "Number of Tweets")
df['year'] = df['date'].dt.year

df['month'] = df['date'].dt.month

df['day'] = df['date'].dt.day

df['dayofweek'] = df['date'].dt.dayofweek

df['hour'] = df['date'].dt.hour

df['dateonly'] = df['date'].dt.date
figure(figsize=(20,6))

sns.set_style("whitegrid")



agg_df = df.groupby(["dateonly"])["text"].count().reset_index()

agg_df.columns = ["dateonly", "count"]



ax = sns.lineplot(x=agg_df["dateonly"], y=agg_df['count'], data=agg_df)

plt.xticks(rotation=90)

ax.set(title="Tweet Count",xlabel="Date",ylabel="Count")



plt.show()
plot_count("dayofweek", df, "Number of Tweets by Day", "Day", "Count")
plot_count("hour", df, "Number of Tweets by Hour", "Hour", "Count")
figure(figsize=(20,6))

miss = pd.DataFrame(df.isnull().sum())



ax = sns.barplot(miss[0], miss.index)

ax.set(title="Missing Data", xlabel="Number of Missing Data")

plt.show()