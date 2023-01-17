import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-whitegrid')

sns.set_style('whitegrid')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
anime=pd.read_csv('/kaggle/input/anime-recommendations-database/anime.csv')

rating=pd.read_csv('/kaggle/input/anime-recommendations-database/rating.csv')
def first_look(df):

    print('dataset shape: \n')

    print('number of rows: ',df.shape[0],' number of columns: ',df.shape[1])

    print('dataset column names: \n')

    print(df.columns)

    print('columns data-type')

    print(df.dtypes)

    print('missing data')

    c=df.isnull().sum()

    print(c[c>0])
first_look(anime)
first_look(rating)
anime['episodes']=anime['episodes'].replace('Unknown',np.nan)

anime['episodes']=anime['episodes'].astype(float)
shared_id=anime[anime['anime_id'].isin(rating['anime_id'])]

shared_id['rating'].isnull().sum()
for i,j in zip(shared_id[shared_id['rating'].isnull()].index,shared_id[shared_id['rating'].isnull()]['anime_id'].values):

    median_value=rating[rating['anime_id']==j]['rating'].median()

    print('median value: ',median_value)

    anime.loc[i,'rating']=median_value

    print('index {} done!'.format(str(i)))
anime.dropna(subset=['rating'],axis=0,inplace=True)
anime['genre']=anime['genre'].str.replace(', ',',')
anime=anime.drop_duplicates('name')
anime['type'].value_counts().plot.pie(autopct='%.1f%%',labels=None,shadow=True,figsize=(8,8))

plt.title('type of Animes in dataset')

plt.ylabel('')

plt.legend(anime['type'].value_counts().index.tolist(),loc='upper right')

plt.show()
plt.figure(figsize=(10,5))

sns.boxplot(x='type',y='rating',data=anime)

plt.title('anime-type VS rating')

plt.show()
for i in anime['type'].unique().tolist():

    print('mean of '+str(i)+' :\n')

    print(anime[anime['type']==i]['rating'].mean())
TV_anime=anime[anime['type']=='TV']

TV_anime['genre'].value_counts().sort_values(ascending=True).tail(20).plot.barh(figsize=(8,8))

plt.title('genres of TV-Animes')

plt.xlabel('frequency')

plt.ylabel('genres')

plt.show()
TV_anime.drop('anime_id',axis=1).describe()
TV_anime[TV_anime['episodes']==TV_anime['episodes'].max()]
TV_anime[TV_anime['episodes']==TV_anime['episodes'].min()]
TV_anime[TV_anime['rating']==TV_anime['rating'].max()]
TV_anime[TV_anime['rating']==TV_anime['rating'].min()]
TV_anime[TV_anime['members']==TV_anime['members'].max()]
TV_anime[TV_anime['members']==TV_anime['members'].min()]
fig=plt.figure(figsize=(13,5))

for i,j in zip(TV_anime[['rating','members']].columns,range(3)):

    ax=fig.add_subplot(1,2,j+1)

    sns.distplot(TV_anime[i],ax=ax)

    plt.axvline(TV_anime[i].mean(),label='mean',color='blue')

    plt.axvline(TV_anime[i].median(),label='median',color='green')

    plt.axvline(TV_anime[i].std(),label='std',color='red')

    plt.title('{} distribtion'.format(i))

    plt.legend()

plt.show()
fig=plt.figure(figsize=(13,5))

for i,j in zip(TV_anime[['rating','members']].columns,range(3)):

    ax=fig.add_subplot(1,2,j+1)

    sns.boxplot(i,data=TV_anime,ax=ax)

    plt.title('{} distribtion'.format(i))

plt.show()
import json

stats=TV_anime.drop('anime_id',axis=1).describe()

def show_outliers(df,col): 

    outliers={}

    for j,k in zip(df[col].index,df[col].tolist()):

        iqr=stats.loc['75%',col]-stats.loc['25%',col]

        upper_bound=stats.loc['75%',col]+iqr*1.5

        lower_bound=stats.loc['25%',col]-iqr*1.5

        if k>upper_bound :

            outliers[k]=['upper',df.loc[j,'name'],df.loc[j,'genre']]

        elif k<lower_bound:

            outliers[k]=['lower',df.loc[j,'name'],df.loc[j,'genre']]

    outliers=json.dumps(outliers)        

    print(outliers)

for i in TV_anime[['rating']].columns:

    print(i)

    print('-'*10)

    show_outliers(TV_anime,i)
iqr=stats.loc['75%','episodes']-stats.loc['25%','episodes']

upper_bound=stats.loc['75%','episodes']+iqr*1.5

lower_bound=stats.loc['25%','episodes']-iqr*1.5

episodes_lst=[]

for i in TV_anime['episodes'].values:

    if i<lower_bound:

        episodes_lst.append('small')

    elif i>upper_bound:

        episodes_lst.append('large')

    elif (i>lower_bound) and (i<upper_bound):

        episodes_lst.append('in-between')

    else:

        episodes_lst.append('no info!')

TV_anime['episodes_classification']=episodes_lst
TV_anime.head()
mean_lst=[]

mean_lst.append(TV_anime[TV_anime['episodes_classification']=='in-between']['rating'].mean())

mean_lst.append(TV_anime[TV_anime['episodes_classification']=='small']['rating'].mean())

mean_lst.append(TV_anime[TV_anime['episodes_classification']=='large']['rating'].mean())

mean_lst.append(TV_anime[TV_anime['episodes_classification']=='no info!']['rating'].mean())

plt.bar(['in-between','small','large','no info!'],mean_lst)

plt.title('mean comparison based on episodes classification')

plt.xlabel('episodes_classification')

plt.ylabel('average-rating')

plt.show()
TV_anime.drop('episodes_classification',axis=1,inplace=True)
TV_animes_df=TV_anime.copy()

TV_animes_df['genre']=TV_animes_df['genre'].str.split(',')

TV_animes_df.head()
for index, lst in zip(TV_animes_df.index,TV_animes_df['genre'].values):

    for genre in lst:

        TV_animes_df.at[index, genre] = 1

#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
TV_animes_df = TV_animes_df.fillna(0)

TV_animes_df.head()
user_input=pd.DataFrame([{'name':'Fullmetal Alchemist: Brotherhood','user_rating':8.6},

                        {'name':'Tokyo Ghoul','user_rating':8}])

user_input
inputId = TV_anime[TV_anime['name'].isin(user_input['name'].tolist())]

user_input = pd.merge(inputId, user_input)

user_input = user_input.drop('genre', 1).drop('rating', 1).drop('episodes',1).drop('type',1).drop('members',1)

user_input
user_anime = TV_animes_df[TV_animes_df['name'].isin(user_input['name'].tolist())]

user_anime=user_anime.drop('rating',1)

user_anime
user_anime = user_anime.reset_index(drop=True)

#Dropping unnecessary issues due to save memory and to avoid issues

user_genre_table = user_anime.drop('anime_id', 1).drop('name', 1).drop('genre', 1).drop('type', 1).drop('episodes',1).drop('members',1)

user_genre_table
userProfile = user_genre_table.transpose().dot(user_input['user_rating'])

userProfile
genre_table = TV_animes_df.set_index(TV_animes_df['anime_id'])

genre_table = genre_table.drop('anime_id', 1).drop('name', 1).drop('genre', 1).drop('episodes', 1).drop('members',1).drop('rating',1).drop('type',1)

genre_table.head()
recommendation_table_df = ((genre_table*userProfile).sum(axis=1))/(userProfile.sum())

recommendation_table_df.head()
recommendation_table_df = recommendation_table_df.sort_values(ascending=False)

#Just a peek at the values

recommendation_table_df.head()
TV_anime.loc[TV_anime['anime_id'].isin(recommendation_table_df.head(10).keys())]