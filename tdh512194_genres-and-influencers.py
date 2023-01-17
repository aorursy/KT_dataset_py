import pandas as pd
import numpy as np
data_movies = pd.read_csv('../input/tmdb_5000_movies.csv')
data_credits = pd.read_csv('../input/tmdb_5000_credits.csv')
pd.set_option('display.max_columns',None)
data_movies.head(5)
from pandas.io.json import json_normalize
import json
def json_decode(data,key):
    result = []
    data = json.loads(data) #convert to jsonjsonn from string
    for item in data: #convert to list from json
        result.append(item[key])
    return result
data_movies.describe(include='all')
data_credits.describe(include='all')
def nan_clean(data,replace=False,alter=''):
    nan_count = len(data) - data.count()
    if np.logical_and(replace,nan_count > 0):
        data.fillna(alter,inplace=True)
        print('Replaced NaN with {}'.format(alter))
        print('Number of cleaned NANs:{}'.format(nan_count))
    else:
        print('Number of NANs:{}'.format(nan_count))
    return 
nan_clean(data_movies.homepage,replace=True)
nan_clean(data_movies.release_date,replace=True)
nan_clean(data_movies.overview,replace=True)
nan_clean(data_movies.runtime,replace=True,alter=0)
nan_clean(data_movies.tagline,replace=True)
data_movies.production_countries = data_movies.production_countries.apply(json_decode,key='name')
data_movies.production_countries.head()
movie_top = data_movies.nlargest(100,'revenue')[['title','revenue','production_countries']]
movie_top
from collections import defaultdict
import pprint
country_top = defaultdict(int)
for data in movie_top.production_countries:
    for item in data:
        country_top[item] += 1
pprint.pprint(country_top)
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.Series(dict(country_top),name='count')
df.index.name='country'
df.plot(kind='barh',grid=True,title='Occurences in the top-100 list')
country_avg_rvn = defaultdict(int)
for index, row in movie_top.iterrows():
    for item in row['production_countries']:
        country_avg_rvn[item] += row['revenue']
for key in country_avg_rvn:
    country_avg_rvn[key] = country_avg_rvn[key]/country_top[key]
pprint.pprint(country_avg_rvn)
df = pd.Series(dict(country_avg_rvn),name='avg_rvn')
df.index.name='country'
df.plot(kind='barh',grid=True,title="Avg revenue per movie in the top-100 for each country")
count = 0
for item in data_movies.production_countries:
    if 'United States of America' in item:
        count +=1
print("The USA produces {0:.0f}% of the movies".format((count/len(data_movies.production_countries))*100))
country_t = defaultdict(int)
for data in data_movies.production_countries:
    for item in data:
        country_t[item] += 1
country_top = dict()
for key in country_t:
    if country_t[key] > 10:
        country_top[key] = country_t[key]
print('List of production countries that produce more than 10 movies:')
pprint.pprint(country_top)
country_avg_rvn = defaultdict(int)
for index, row in data_movies.iterrows():
    for item in row['production_countries']:
        if item in list(country_top.keys()):
            country_avg_rvn[item] += row['revenue']
for key in country_avg_rvn:
    country_avg_rvn[key] = country_avg_rvn[key]/country_top[key]
pprint.pprint(dict(country_avg_rvn))
df = pd.Series(dict(country_avg_rvn),name='avg_rvn')
df.index.name='country'
df.plot(kind='bar',grid=True,title="Avg revenue per movie in original list for each country",figsize=(10,5))
data_movies.genres = data_movies.genres.apply(json_decode,key='name')
data_movies.genres
genres = set()
for item in data_movies.genres:
    for genre in item:
        genres.add(genre)
genres = list(genres)
genres.append('revenue')
print(genres)
df = pd.DataFrame(columns=genres)
for index, row in data_movies.iterrows():
    for item in row['genres']:
        df.loc[index,item] = 1
    df.loc[index,'revenue'] = row['revenue']
df.fillna(0,inplace=True)
df.head()
df.revenue = (df.revenue - df.revenue.mean())/df.revenue.std()
Y = np.array(df.revenue)
Y
x1 = np.ones(len(df)).reshape(len(df),1)
x2 = df.iloc[:,:-1].as_matrix()
X = np.concatenate((x1,x2),axis=1)
X
W = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
pprint.pprint(list(W))
genres_d = genres[:-1]
weights = dict(zip(genres_d, W[1:]))
weights
df = pd.Series(weights,name='genre_rvn_weight')
df.index.name='genre'
df.plot(kind='bar',grid=True,title="Weights of genres on the revenue",figsize=(15,5))
del genres[-1]
genres.append('avg_vote')
genres
df = pd.DataFrame(columns=genres)
for index, row in data_movies.iterrows():
    for item in row['genres']:
        df.loc[index,item] = 1
    df.loc[index,'avg_vote'] = row['vote_average']
df.fillna(0,inplace=True)
df.head()
df.avg_vote = (df.avg_vote - df.avg_vote.mean())/df.avg_vote.std()
Y = np.array(df.avg_vote)
Y
W = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
pprint.pprint(list(W))
genres_d = genres[:-1]
weights = dict(zip(genres_d, W[1:]))
weights
df = pd.Series(weights,name='genre_avgvote_weight')
df.index.name='genre'
df.plot(kind='bar',grid=True,title="Weights of genres on the avg_vote",figsize=(15,5))
import datetime
string_date = data_movies.release_date[0]
datetime.datetime.strptime(string_date,"%Y-%m-%d").isoweekday()
weekdays = {
    1 : 'Mon',
    2 : 'Tue',
    3 : 'Wed',
    4 : 'Thu',
    5 : 'Fri',
    6 : 'Sat',
    7 : 'Sun'
}
def to_weekday(string_date):
    if string_date != '':
        weekday = datetime.datetime.strptime(string_date,"%Y-%m-%d").isoweekday()
        return weekdays[weekday]
    else:
        return np.nan
data_movies['release_weekday'] = data_movies.release_date.apply(to_weekday)
data_movies.release_weekday.value_counts(dropna=False)
df = data_movies[pd.notnull(data_movies['release_weekday'])] #omit the null weekdays
df = df.loc[:,['revenue','release_weekday']] 
df = df[df['revenue']!=0] #omit the zero revenues
df
#reset index to join to dataframe
mon = df[df['release_weekday']=='Mon'].sample(100).reset_index()
tue = df[df['release_weekday']=='Tue'].sample(100).reset_index()
wed = df[df['release_weekday']=='Wed'].sample(100).reset_index()
thu = df[df['release_weekday']=='Thu'].sample(100).reset_index()
fri = df[df['release_weekday']=='Fri'].sample(100).reset_index()
sat = df[df['release_weekday']=='Sat'].sample(100).reset_index()
sun = df[df['release_weekday']=='Sun'].sample(100).reset_index()
df = pd.DataFrame({
    'Mon':mon['revenue'],
    'Tue':tue['revenue'],
    'Wed':wed['revenue'],
    'Thu':thu['revenue'],
    'Fri':fri['revenue'],
    'Sat':sat['revenue'],
    'Sun':sun['revenue']
})
df
import scipy.stats as stats
F,p = stats.f_oneway(
    df['Mon'],
    df['Tue'],
    df['Wed'],
    df['Thu'],
    df['Fri'],
    df['Sat'],
    df['Sun']
)
F,p
weekend = pd.concat([fri, sat, sun]).sample(30).reset_index()
other = pd.concat([mon, tue, wed, thu]).sample(30).reset_index()
weekend
other
df = pd.DataFrame({
    'Weekend':weekend['revenue'],
    'Other':other['revenue']
})
df
F,p = stats.f_oneway(
    df['Other'],
    df['Weekend'],
)
F, p
df = data_movies[pd.notnull(data_movies['release_date'])] #omit the null weekdays
df = df.loc[:,['revenue','release_date']] 
df = df[df['revenue']!=0] #omit the zero revenues
df['release_date'] = pd.to_datetime(df.release_date)

def to_month(date):
    return date.month

df['release_date'] = df.release_date.apply(to_month)

jan = df[df['release_date'] ==1].sample(30).reset_index()
feb = df[df['release_date'] ==2].sample(30).reset_index()
mar = df[df['release_date'] ==3].sample(30).reset_index()
apr = df[df['release_date'] ==4].sample(30).reset_index()
may = df[df['release_date'] ==5].sample(30).reset_index()
jun = df[df['release_date'] ==6].sample(30).reset_index()
jul = df[df['release_date'] ==7].sample(30).reset_index()
aug = df[df['release_date'] ==8].sample(30).reset_index()
sep = df[df['release_date'] ==9].sample(30).reset_index()
oct_ = df[df['release_date'] ==10].sample(30).reset_index()
nov = df[df['release_date'] ==11].sample(30).reset_index()
dec = df[df['release_date']==12].sample(30).reset_index()

df = pd.DataFrame({
    'Jan':jan['revenue'],
    'Feb':feb['revenue'],
    'Mar':mar['revenue'],
    'Apr':apr['revenue'],
    'May':may['revenue'],
    'Jun':jun['revenue'],
    'Jul':jul['revenue'],
    'Aug':aug['revenue'],
    'Sep':sep['revenue'],
    'Oct':oct_['revenue'],
    'Nov':nov['revenue'],
    'Dec':dec['revenue']
})

F,p = stats.f_oneway(
    df['Jan'],
    df['Feb'],
    df['Mar'],
    df['Apr'],
    df['May'],
    df['Jun'],
    df['Jul'],
    df['Aug'],
    df['Sep'],
    df['Oct'],
    df['Nov'],
    df['Dec']
)

F,p
df = data_movies[pd.notnull(data_movies['release_date'])] #omit the null weekdays
df = df.loc[:,['revenue','release_date']] 
df = df[df['revenue']!=0] #omit the zero revenues
df['release_date'] = pd.to_datetime(df.release_date)

def to_month(date):
    return date.month

df['release_date'] = df.release_date.apply(to_month)

jan = df[df['release_date'] ==1].revenue.mean()
feb = df[df['release_date'] ==2].revenue.mean()
mar = df[df['release_date'] ==3].revenue.mean()
apr = df[df['release_date'] ==4].revenue.mean()
may = df[df['release_date'] ==5].revenue.mean()
jun = df[df['release_date'] ==6].revenue.mean()
jul = df[df['release_date'] ==7].revenue.mean()
aug = df[df['release_date'] ==8].revenue.mean()
sep = df[df['release_date'] ==9].revenue.mean()
oct_ = df[df['release_date'] ==10].revenue.mean()
nov = df[df['release_date'] ==11].revenue.mean()
dec = df[df['release_date']==12].revenue.mean()
month_avg_rvn = {
    1:jan,
    2:feb,
    3:mar,
    4:apr,
    5:may,
    6:jun,
    7:jul,
    8:aug,
    9:sep,
    10:oct_,
    11:nov,
    12:dec
}
df = pd.Series(dict(month_avg_rvn),name='avg_rvn')
df.index.name='month'
df.plot(kind='bar',grid=True,title="Avg revenue per movie in original list for each month",figsize=(10,5))
df = data_movies.loc[:,['revenue','release_date']]
df['release_date'] = pd.to_datetime(df.release_date)
df = df.sort_values('release_date')
df = df[df['revenue']!=0]
df = df.set_index('release_date')
df
df.plot(grid=True,figsize=(20,10,),title='Revenue over the period').set_ylabel('Revenue')
