import pandas as pd

from matplotlib.pyplot import imshow, subplots
df=pd.read_csv('/kaggle/input/biden-and-trump-tweets-201720/biden_trump_tweets.csv',dtype={'id':str,'username':str,'hour_utc':int, 'minute_utc':int}, parse_dates=['date_utc'])

df.head()
unique_users =  df.username.unique();

print(f'Unique Users ({len(unique_users)}): {unique_users}')

for user in unique_users:

    n_tweets = df[df.username==user].id.nunique()

    first_tweet = df[df.username==user].date_utc.min().strftime("%H:%M %d %b %Y")

    last_tweet = df[df.username==user].date_utc.max().strftime("%H:%M %d %b %Y")

    print(f'\tUser @{user} has {n_tweets:,} tweets between {first_tweet} and {last_tweet}')
all_hours=[x for x in range(0,24)]

all_mins =[x for x in range(0,60)]



donald_matrix = df[df.username=='realDonaldTrump'].groupby(['hour_utc','minute_utc']).id.count().reset_index().pivot('hour_utc','minute_utc','id')

donald_matrix = donald_matrix.reindex(all_hours,axis=0).reindex(all_mins,axis=1).fillna(0).astype(int)



joe_matrix = df[df.username=='JoeBiden'].groupby(['hour_utc','minute_utc']).id.count().reset_index().pivot('hour_utc','minute_utc','id')

joe_matrix = joe_matrix.reindex(all_hours,axis=0).reindex(all_mins,axis=1).fillna(0).astype(int)

fig,(ax1,ax2) = subplots(1,2,figsize=(16,8))

ax1.imshow(donald_matrix); ax2.imshow(joe_matrix)

ax1.set_title('@realDonaldTrump\n'); ax2.set_title('@JoeBiden\n'); 

ax1.set_xlabel('minute'); ax1.set_ylabel('hour'); ax2.set_xlabel('minute'); ax2.set_ylabel('hour');