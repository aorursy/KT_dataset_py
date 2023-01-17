import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from google.colab import drive

drive.mount('/content/gdrive')
transaction = pd.read_csv("/content/gdrive/My Drive/MBA_class/preprogression/transactions.csv")

mem_data = pd.read_csv("/content/gdrive/My Drive/MBA_class/preprogression/mem_data.csv")

songs = pd.read_csv("/content/gdrive/My Drive/MBA_class/preprogression/songs.csv")
# songs = pd.read_csv("data/songs.csv")

# mem_data = pd.read_csv('data/mem_data.csv')

# transaction = pd.read_csv('data/transactions.csv')
songs.head()
mem_data.head()
transaction.head()
sns.boxplot(x= 'gender', y= 'reg_date', data=mem_data);
sns.boxplot(x= 'gender', y= 'ex_date', data=mem_data);
all_data = transaction.merge(mem_data, how='left', on='user_id').merge(songs, how='left', on='song_id')
member_feature = mem_data[['user_id', 'gender']].copy()
song_feature = songs.copy()
member_feature
all_data
all_data.isnull().sum()
f = pd.to_datetime(mem_data.reg_date, format='%Y%m%d')

f = (pd.to_datetime('2017-12-31') - f).dt.days

mem_data['E_DAY'] = f

mem_data.E_DAY.describe()
f = transaction.groupby('user_id')['listen'].agg({'total_listen':'sum'}).reindex().reset_index()

member_feature = mem_data.merge(f, how='left')

member_feature.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0)
f = transaction.merge(songs, how='left')

f = f.groupby('user_id')['genre'].agg({'rec_genre':'nunique'}).reindex().reset_index()

member_feature = mem_data.merge(f, how='left')

member_feature.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0)
f = all_data.groupby('user_id')['isrc'].agg([('song_concentration', 'value_counts')]).reset_index()

f= f.groupby('user_id')['song_concentration'].agg([('song_concent', lambda x: x.max()/x.sum())]).reset_index()
k = f.where(f.song_concent>0, other=0)

k = np.log(k.song_concent +1)

f.song_concent=k
plt.hist(f.song_concent);
member_feature= member_feature.merge(f, how='left', on='user_id')
f = all_data.groupby('user_id')['genre'].agg([('genre_concentration', 'value_counts')]).reset_index()

f= f.groupby('user_id')['genre_concentration'].agg([('genre_concent', lambda x: x.max()/x.sum())]).reset_index()
k = f.where(f.genre_concent>0, other=0)

k = np.log(k.genre_concent +1)

f.genre_concent=k

plt.hist(f.genre_concent);
member_feature= member_feature.merge(f, how='left', on='user_id'); member_feature
f = all_data.groupby('user_id')['artist'].agg([('artist_concentration', 'value_counts')]).reset_index()

f= f.groupby('user_id')['artist_concentration'].agg([('fanship', lambda x: x.max()/x.sum())]).reset_index()
k = f.where(f.fanship>0, other=0)

k = np.log(k.fanship +1)

f.fanship=k

plt.hist(f.fanship);
member_feature= member_feature.merge(f, how='left', on='user_id'); member_feature
f = all_data.groupby('user_id')['composer'].agg([('composer_concentration', 'value_counts')]).reset_index()

f= f.groupby('user_id')['composer_concentration'].agg([('composer_concent', lambda x: x.max()/x.sum())]).reset_index()
k = f.where(f.composer_concent>0, other=0)

k = np.log(k.composer_concent +1)

f.composer_concent=k

plt.hist(f.composer_concent);
member_feature= member_feature.merge(f, how='left', on='user_id'); member_feature
f = all_data.groupby('user_id')['lyricist'].agg([('lyricist_concentration', 'value_counts')]).reset_index()

f= f.groupby('user_id')['lyricist_concentration'].agg([('lyricist_concent', lambda x: x.max()/x.sum())]).reset_index()
k = f.where(f.lyricist_concent>0, other=0)

k = np.log(k.lyricist_concent +1)

f.lyricist_concent=k

plt.hist(f.lyricist_concent);
member_feature= member_feature.merge(f, how='left', on='user_id'); member_feature


contract= mem_data[['user_id', 'reg_date','ex_date']]



contract['reg_date'] = pd.to_datetime(contract.reg_date, format='%Y%m%d')

contract['ex_date'] = pd.to_datetime(contract.ex_date, format='%Y%m%d')



contract['contract_period'] = (contract.ex_date - contract.reg_date).dt.days
f = contract.drop(['reg_date', 'ex_date'], axis=1)

plt.hist(contract.contract_period);
member_feature = member_feature.merge(f, how='left', on = 'user_id')
mem_data.reg_method.value_counts()
f = mem_data[['user_id', 'reg_method']] 

f = pd.get_dummies(f, columns=['reg_method'])

 

member_feature = member_feature.merge(f, how='left', on='user_id')
f = mem_data[['user_id']]

f['reg_year'] = pd.to_datetime(mem_data.reg_date, format='%Y%m%d').dt.year

member_feature = member_feature.merge(f, how='left', on='user_id')
k = all_data.groupby(['user_id', 'rec_loc'])['listen'].agg([('listen_YN', 'sum')]).reset_index();
f = k.groupby('user_id')['listen_YN', 'rec_loc'].max().reset_index()

f = pd.get_dummies(f)

f= f.drop('listen_YN', axis=1)

member_feature= member_feature.merge(f, how='left', on='user_id')

f
f1 = k.groupby('user_id')['listen_YN'].agg([('res_listen_all', 'sum')]).reset_index()

f2 = k.groupby('user_id')['listen_YN'].agg([('res_listen_max_v', 'max')]).reset_index()



f = f1.merge(f2, how='left', on='user_id')

f['rec_loc_concent'] = f.res_listen_max_v / f.res_listen_all

f= f.drop(['res_listen_all', 'res_listen_max_v'], axis=1)



member_feature = member_feature.merge(f, how='left', on='user_id')
k = all_data.groupby(['user_id', 'rec_screen'])['listen'].agg([('listen_YN', 'sum')]).reset_index();
f = k.groupby('user_id')['listen_YN', 'rec_screen'].max().reset_index()

f = pd.get_dummies(f, columns=['rec_screen'])

f = f.drop('listen_YN', axis=1)

member_feature= member_feature.merge(f, how='left', on='user_id')
f1 = k.groupby('user_id')['listen_YN'].agg([('res_listen_all', 'sum')]).reset_index()

f2 = k.groupby('user_id')['listen_YN'].agg([('res_listen_max_v', 'max')]).reset_index()



f = f1.merge(f2, how='left', on='user_id')

f['rec_screen_concent'] = f.res_listen_max_v / f.res_listen_all

f= f.drop(['res_listen_all', 'res_listen_max_v'], axis=1)



member_feature = member_feature.merge(f, how='left', on='user_id')
f = all_data.groupby('user_id')['listen'].agg({'rec_listen_sum' : 'sum'}).reset_index(); f.head()
member_feature= member_feature.merge(f, how='left', on='user_id')
age= mem_data[['user_id', 'age']]
age.age[age.age>90] = np.mean(age.age)
age.age[age.age<5] = np.mean(age.age)
sns.boxplot(age.age);
def age_about(x):

    if x<=15: 

        return('0~15')

    elif 15<x<=25:

        return('16~25')

    elif 25<x<=35:

        return('26~35')

    elif 35<x<=45:

        return('36~45')

    elif 45<x<=55:

        return('46~55')

    elif 55<x<=65:

        return('56~65')

    elif 65<x:

        return('66~')



age['age_about'] = age.age.apply(age_about)
f = pd.get_dummies(age)

member_feature = member_feature.merge(f, how='left', on='user_id')
f = all_data.groupby(['user_id','language'])['language'].agg([('language_count', 'size')]).reset_index(); f.head()
f2 = f.groupby('user_id')['language'].agg([('language_viriable', 'size')]).reset_index(); f2.head()
member_feature = member_feature.merge(f2, how='left', on='user_id')
f3 = all_data.groupby('user_id')['language'].agg([('prefer_language', lambda x: x.value_counts().index[0])]).reset_index()

f3 = pd.get_dummies(f3, columns=['prefer_language'])
member_feature = member_feature.merge(f3, how='left', on='user_id')
f = all_data.groupby('user_id')['genre'].agg([('prefer_genre', lambda x: x.value_counts().index[0])]).reset_index()

member_feature = member_feature.merge(f, how='left', on='user_id')
f = all_data.groupby('user_id')['artist'].agg([('prefer_artist', lambda x: x.value_counts().index[0])]).reset_index()

member_feature = member_feature.merge(f, how='left', on='user_id')
f = all_data.groupby('user_id')['composer'].agg([('prefer_composer', lambda x: x.value_counts().index[0])]).reset_index()

member_feature = member_feature.merge(f, how='left', on='user_id')
f = all_data.groupby('user_id')['lyricist'].agg([('prefer_lyricist', lambda x: x.value_counts().index[0])]).reset_index()

member_feature = member_feature.merge(f, how='left', on='user_id')
f = all_data.groupby('user_id')['song_id'].agg([('prefer_song', lambda x: x.value_counts().index[0])]).reset_index()

member_feature = member_feature.merge(f, how='left', on='user_id')
f = pd.pivot_table(data = all_data, index='isrc', columns='gender', aggfunc='size').reset_index()

f['female_ratio_song'] = f.female/(f.female+f.male)
sns.boxplot(y= f.female_ratio_song);
plt.hist(f.female_ratio_song);
f.female_ratio_song.quantile(q=0.25)
f.female_ratio_song.quantile(q=0.75)
def gender_prefer(x):

    if x < f.female_ratio_song.quantile(q=0.20):

        return 'male_prefer'

    elif x > f.female_ratio_song.quantile(q=0.80):

        return 'female_prefer'

    else : 

        return 'everyone_like'



    

f['song_gender_prefer'] = f.female_ratio_song.apply(gender_prefer)
f.song_gender_prefer.value_counts()
f.columns
f= f.drop(['female', 'male', 'unknown', 'female_ratio_song'], axis=1)
song_feature = song_feature.merge(f, how='left', on='isrc')
f = pd.pivot_table(data = all_data, index='composer', columns='gender', aggfunc='size').reset_index()

f['female_ratio_composer'] = f.female/(f.female+f.male)
sns.boxplot(y= f.female_ratio_composer);
def gender_prefer(x):

    if x < f.female_ratio_composer.quantile(q=0.20):

        return 'male_prefer'

    elif x > f.female_ratio_composer.quantile(q=0.80):

        return 'female_prefer'

    else : 

        return 'everyone_like'



    

f['composer_gender_prefer'] = f.female_ratio_composer.apply(gender_prefer)

plt.bar(f.composer_gender_prefer.unique(), f.composer_gender_prefer.value_counts());
f.columns
f= f.drop(['female', 'male', 'unknown', 'female_ratio_composer'], axis=1)
song_feature = song_feature.merge(f, how='left', on='composer')
f = pd.pivot_table(data = all_data, index='lyricist', columns='gender', aggfunc='size').reset_index()

f['female_ratio_lyricist'] = f.female/(f.female+f.male)
sns.boxplot(y= f.female_ratio_lyricist);
def gender_prefer(x):

    if x < f.female_ratio_lyricist.quantile(q=0.20):

        return 'male_prefer'

    elif x > f.female_ratio_lyricist.quantile(q=0.80):

        return 'female_prefer'

    else : 

        return 'everyone_like'



    

f['lyricist_gender_prefer'] = f.female_ratio_lyricist.apply(gender_prefer)

plt.bar(f.lyricist_gender_prefer.unique(), f.lyricist_gender_prefer.value_counts());
f.columns
f= f.drop(['female', 'male', 'unknown', 'female_ratio_lyricist'], axis=1)
song_feature = song_feature.merge(f, how='left', on='lyricist')
f = pd.pivot_table(data = all_data, index='artist', columns='gender', aggfunc='size').reset_index()

f['female_ratio_artist'] = f.female/(f.female+f.male)
#sns.boxplot(y= f.female_ratio_artist);
def gender_prefer(x):

    if x < f.female_ratio_artist.quantile(q=0.20):

        return 'male_prefer'

    elif x > f.female_ratio_artist.quantile(q=0.80):

        return 'female_prefer'

    else : 

        return 'everyone_like'



    

f['artist_gender_prefer'] = f.female_ratio_artist.apply(gender_prefer)

#plt.bar(f.artist_gender_prefer.unique(), f.artist_gender_prefer.value_counts());
f.columns
f= f.drop(['female', 'male', 'unknown', 'female_ratio_artist'], axis=1)
song_feature = song_feature.merge(f, how='left', on='artist')
f = all_data.groupby('artist')['genre'].agg([('artist_genre_var',lambda x: x.nunique())]).reset_index()

song_feature = song_feature.merge(f, how='left', on='artist')
f = all_data.groupby('composer')['genre'].agg([('composer_genre_var',lambda x: x.nunique())]).reset_index()

song_feature = song_feature.merge(f, how='left', on='composer')
f = all_data.groupby('lyricist')['genre'].agg([('lyricist_genre_var',lambda x: x.nunique())]).reset_index()

song_feature = song_feature.merge(f, how='left', on='lyricist')
f = all_data.groupby('isrc')['artist', 'composer', 'lyricist'].mean().reset_index() 
def dual_more(x,y,z):

  if x==y!=z:

    return('artist_composer')

  elif x==y==z:

    return('artist_all')

  elif x==z!=y:

    return('artist_lyricist')

  else : 

    return('only_artist')

  

f2 =[]

for i in range(len(f.isrc)):

  f2.append(dual_more(f.artist[i], f.composer[i], f.lyricist[i]))

  

f['roll'] = f2

dropper = ['artist', 'composer', 'lyricist']

f = f.drop(dropper, axis=1) ; f.head()
song_feature = song_feature.merge(f, how='left', on='isrc')
k = all_data.listen.sum()

f2 = all_data.groupby(['artist','isrc'])['listen'].agg([('listned', 'sum')]).reset_index()

f2['listened_rate'] = f2.listned / k

f = f2.groupby('artist')['listened_rate'].agg([('popular', 'sum')]).reset_index()



k= f.popular.where(f.popular >0, other=0)

k= np.log(k+1)

f.popular = k



plt.hist(f.popular);
song_feature= song_feature.merge(f, how='left', on='artist')
f = all_data[['artist','isrc']]
k =[]



for i in range(len(f.isrc)): 

  k.append(f['isrc'][i][5:7])



f['publish_year'] = k
def by_year(x): 

  if int(x) < 18:

    return(int('20'+x))

  else : 

    return(int('19'+x))



f.publish_year = f.publish_year.apply(by_year)
sns.boxplot(f.publish_year);
f2 = f.groupby('artist')['publish_year'].agg([('debut_year', 'min')]).reset_index()
sns.boxplot(f2.debut_year);
f = f.merge(f2, how='left', on='artist')
song_feature = song_feature.merge(f, how='left', on='isrc')
song_feature.head()
song_feature.columns
song_feature = song_feature.drop('isrc', axis=1)
song_feature = song_feature.rename(columns = {'artist_x': 'artist'})
song_feature.columns
song_feature = song_feature.drop_duplicates()
f = song_feature.groupby('song_id')['song_gender_prefer', 'publish_year'].max().reset_index()
f2 = f.rename(columns={'song_id' : 'prefer_song'}).copy()

f2 = pd.get_dummies(f2, columns=['song_gender_prefer'])
member_feature= member_feature.merge(f2, how='left', on='prefer_song')
f3 = pd.get_dummies(f, columns=['song_gender_prefer'])
all_data = all_data.merge(f3, how='left', on='song_id')
f3 = all_data.groupby('user_id')[['song_gender_prefer_everyone_like', 'song_gender_prefer_female_prefer', 'song_gender_prefer_male_prefer']].sum().reset_index()

f3['song_everyone_like_rate'] = f3.song_gender_prefer_everyone_like/(f3.drop('user_id', axis=1).sum(axis=1))

f3['song_male_prefer_rate'] = f3.song_gender_prefer_male_prefer/(f3.drop('user_id', axis=1).sum(axis=1))

f3['song_female_prefer_rate'] = f3.song_gender_prefer_female_prefer/(f3.drop('user_id', axis=1).sum(axis=1))

f3 = f3.drop(['song_gender_prefer_everyone_like', 'song_gender_prefer_female_prefer', 'song_gender_prefer_male_prefer'], axis=1)
member_feature = member_feature.merge(f3, how='left', on='user_id')
f = song_feature.groupby('composer')['composer_gender_prefer', 'composer_genre_var'].max().reset_index()
f2 = f.rename(columns={'composer' : 'prefer_composer'}).copy()

f2 = pd.get_dummies(f2, columns=['composer_gender_prefer'])
member_feature = member_feature.merge(f2, how='left', on='prefer_composer')
f3 = pd.get_dummies(f, columns=['composer_gender_prefer'])

all_data = all_data.merge(f3, how='left', on='composer')
f3 = all_data.groupby('user_id')[['composer_gender_prefer_everyone_like', 'composer_gender_prefer_female_prefer', 'composer_gender_prefer_male_prefer']].sum().reset_index()

f3['composer_everyone_like_rate'] = f3.composer_gender_prefer_everyone_like/(f3.drop('user_id', axis=1).sum(axis=1))

f3['composer_male_prefer_rate'] = f3.composer_gender_prefer_male_prefer/(f3.drop('user_id', axis=1).sum(axis=1))

f3['composer_female_prefer_rate'] = f3.composer_gender_prefer_female_prefer/(f3.drop('user_id', axis=1).sum(axis=1))

f3 = f3.drop(['composer_gender_prefer_everyone_like', 'composer_gender_prefer_female_prefer', 'composer_gender_prefer_male_prefer'], axis=1)

member_feature = member_feature.merge(f3, how='left', on='user_id')
f = song_feature.groupby('lyricist')['lyricist_gender_prefer', 'lyricist_genre_var'].max().reset_index()
f2 = f.rename(columns={'lyricist' : 'prefer_lyricist'}).copy()

f2 = pd.get_dummies(f2, columns=['lyricist_gender_prefer'])

member_feature = member_feature.merge(f2, how='left', on='prefer_lyricist')
f3 = pd.get_dummies(f, columns=['lyricist_gender_prefer'])

all_data = all_data.merge(f3, how='left', on='lyricist')
f3 = all_data.groupby('user_id')[['lyricist_gender_prefer_everyone_like', 'lyricist_gender_prefer_female_prefer', 'lyricist_gender_prefer_male_prefer']].sum().reset_index()

f3['lyricist_everyone_like_rate'] = f3.lyricist_gender_prefer_everyone_like/(f3.drop('user_id', axis=1).sum(axis=1))

f3['lyricist_male_prefer_rate'] = f3.lyricist_gender_prefer_male_prefer/(f3.drop('user_id', axis=1).sum(axis=1))

f3['lyricist_female_prefer_rate'] = f3.lyricist_gender_prefer_female_prefer/(f3.drop('user_id', axis=1).sum(axis=1))

f3 = f3.drop(['lyricist_gender_prefer_everyone_like', 'lyricist_gender_prefer_female_prefer', 'lyricist_gender_prefer_male_prefer'], axis=1)

member_feature = member_feature.merge(f3, how='left', on='user_id')
f = song_feature.groupby('artist')['artist_gender_prefer', 'artist_genre_var', 'roll', 'popular', 'debut_year'].max().reset_index()
f2 = f.rename(columns={'artist' : 'prefer_artist'}).copy()

f2 = pd.get_dummies(f2, columns=['artist_gender_prefer', 'roll'])

member_feature = member_feature.merge(f2, how='left', on='prefer_artist')
f3 = pd.get_dummies(f, columns=['artist_gender_prefer'])

all_data = all_data.merge(f3, how='left', on='artist')
f3 = all_data.groupby('user_id')[['artist_gender_prefer_everyone_like', 'artist_gender_prefer_female_prefer', 'artist_gender_prefer_male_prefer']].sum().reset_index()

f3['artist_everyone_like_rate'] = f3.artist_gender_prefer_everyone_like/(f3.drop('user_id', axis=1).sum(axis=1))

f3['artist_male_prefer_rate'] = f3.artist_gender_prefer_male_prefer/(f3.drop('user_id', axis=1).sum(axis=1))

f3['artist_female_prefer_rate'] = f3.artist_gender_prefer_female_prefer/(f3.drop('user_id', axis=1).sum(axis=1))

f3 = f3.drop(['artist_gender_prefer_everyone_like', 'artist_gender_prefer_female_prefer', 'artist_gender_prefer_male_prefer'], axis=1)

member_feature = member_feature.merge(f3, how='left', on='user_id')
member_feature.columns
member_feature = pd.get_dummies(member_feature, columns=['prefer_genre', 'prefer_artist', 'prefer_composer', 'prefer_lyricist', 'prefer_song'])
member_feature.isnull().sum()
member_feature=member_feature.fillna(0)
k=[]



for i in range(len(member_feature.columns)):

  if member_feature.columns[i] != 'user_id' and member_feature.columns[i] != 'gender':

    k.append(member_feature.columns[i])
member_feature[k] = member_feature[k].astype(int)
member_feature.info()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



X_train, X_test, y_train, y_test = train_test_split(member_feature.drop(['user_id','gender'], axis=1), member_feature.gender)

tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)
feats = {}

for feature, importance in zip(X_train.columns, tree.feature_importances_):

    feats[feature] = importance 



importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'importance'})

#

#importances.sort_values(by='importance', ascending=True).plot(kind='barh', figsize=(20,500));
#sns.boxplot(importances.importance);
len(member_feature.columns)
zero_features = importances.index[importances.importance ==0]
member_feature = member_feature.drop(zero_features, axis=1)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



X_train, X_test, y_train, y_test = train_test_split(member_feature.drop(['user_id','gender'], axis=1), member_feature.gender)

tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)
feats = {}

for feature, importance in zip(X_train.columns, tree.feature_importances_):

    feats[feature] = importance 



importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'importance'})



#importances.sort_values(by='importance', ascending=True).plot(kind='barh', figsize=(20,100));
#sns.boxplot(importances.importance);
len(member_feature.columns)
q = 0.7





unnecessary_features = importances.index[importances.importance < importances.importance.quantile(q=q)]

importances.importance.quantile(q=q)
member_feature = member_feature.drop(unnecessary_features, axis=1)
X_train, X_test, y_train, y_test = train_test_split(member_feature.drop(['user_id','gender'], axis=1), member_feature.gender)

tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)
feats = {}

for feature, importance in zip(X_train.columns, tree.feature_importances_):

    feats[feature] = importance 



importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'importance'})

#importances.sort_values(by='importance', ascending=True).plot(kind='barh', figsize=(20,100));
sns.boxplot(importances.importance);
len(member_feature.columns)
member_feature.shape
member_feature.head()
member_feature.to_csv("/content/gdrive/My Drive/MBA_class/preprogression/pre_data.csv")
# sns.pairplot(all_data[all_data.gender!='unknown'], hue='gender');
member_feature.head()
sns.pairplot(member_feature.drop('user_id', axis=1), hue='gender')