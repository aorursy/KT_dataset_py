import pandas as pd
import seaborn as sns
import re
import numpy as np
import datetime
df = pd.read_csv('../input/googleplaystore.csv')
print(df.columns)
print(df.shape)
df.head(2)
bad_filling = df[df.index == 10472]
bad_filling.drop([u'Android Ver'],axis=1,inplace=True)
bad_filling.columns = [u'App', u'Rating', u'Reviews', u'Size', u'Installs',
       u'Type', u'Price', u'Content Rating', u'Genres', u'Last Updated',
       u'Current Ver', u'Android Ver']
bad_filling[u'Category'] = ''
bad_filling = bad_filling[[u'App', u'Category', u'Rating', u'Reviews', u'Size', u'Installs',
       u'Type', u'Price', u'Content Rating', u'Genres', u'Last Updated',
       u'Current Ver', u'Android Ver']]
bad_filling

df = df.drop(10472)
df = pd.concat([df,bad_filling])
df.tail()
df.to_csv('correctedgoogleplaystore.csv',index=False)
df = pd.read_csv('correctedgoogleplaystore.csv')
df.dtypes
def extract_size(size):
    regexp_M = re.match(r'([0-9\.]+)M',size)
    regexp_k = re.match(r'([0-9\.]+)k',size)
    if regexp_M:
        ret = float(regexp_M.groups()[0])*1000
    elif regexp_k:
        ret = float(regexp_k.groups()[0])
    elif size == "variable" or size == "Varies with device":
        ret = np.nan
    else:
        ret = "unknown"
    return ret

df.Size.value_counts()
df['Size_k'] = df['Size'].apply(extract_size)
df.drop(['Size'],axis=1,inplace=True)
df.rename(columns={'Size_k':'Size'}, inplace=True)
df.head(2)
df[["Size"]].dtypes
def convert_price(price):
    dollars = re.match(r'\$([0-9\.,]+)',price)
    if dollars:
        ret = float(dollars.groups()[0])
    elif price == '0':
        ret = float(0)
    else:
        ret = "unknown"
    return ret

df['new_price'] = df['Price'].apply(convert_price)
print(df[["new_price"]].dtypes)
df.drop(['Price'],axis=1,inplace=True)
df.rename(columns={'new_price':'Price'}, inplace=True)
df['Last Updated'] = pd.to_datetime(df['Last Updated'])
df.head(2)
def from_date_to_str(mydate):
    return datetime.datetime.strftime(mydate,"%m %d, %Y")

print('The most recent update was done %s' % df[['Last Updated']].max().apply(from_date_to_str))
print('The most ancient update was done %s' % df[['Last Updated']].min().apply(from_date_to_str))
df['since_last_update'] = (df[['Last Updated']].max()[0] - df['Last Updated']).apply(lambda x : x.days)
df.head(2)
print("Original size = %d" % df.shape[0])
df.App.value_counts().head(2)
df_dropdup = df.copy()
df_dropdup = df_dropdup.drop_duplicates()
print("Size after dropping exact duplicates = %d" % df_dropdup.shape[0])
gb = df_dropdup.groupby('App')
gb = gb.agg({'Reviews' : 'max','since_last_update':'min'}).reset_index()

print("After filter on max reviews and most recent updates, number of different apps = %d" % gb.shape[0])
df_nodup = pd.merge(gb,df_dropdup,on=['App','Reviews','since_last_update'],how='left')
print("Size after filtering on max reviews and most recent updates = %d" % df_nodup.shape[0])
dup = pd.DataFrame(df_nodup.App.value_counts()).reset_index()
dup.columns = ['App','nb']
for a in dup[dup.nb > 1].App:
    tmp = df_nodup[df_nodup.App == a].copy().drop('Category',axis=1).drop_duplicates()
    print('Applications : %s' % (a))
    print('If ignoring the category, number of unique lines = %d' % tmp.shape[0])
    print('Category details')
    print(df_nodup['Category'][df_nodup.App == a].value_counts())
    print('')
df = df_nodup
def extract_genres(genre):
    if ';' in genre:
        ret = genre.split(';')
    else:
        ret = [genre]
    return ret

df_genres = df[['App','Genres']]
df_genres['Genres_tab'] = df_genres['Genres'].fillna('').apply(extract_genres)
df_genres['len_Genres_tab'] = df_genres['Genres_tab'].apply(len)
df_genres.head()
print("The mininmum number of genres for one app is %d" % df_genres.len_Genres_tab.min())
print("The maximum number of genres is %d" % df_genres.len_Genres_tab.max())
print("The median number of genres is %f" % df_genres.len_Genres_tab.median())
print("The mean number of genres is %f" % df_genres.len_Genres_tab.mean())
def extract_2_genres(genre):
    if ';' in genre:
        ret = genre.split(';')
    else:
        ret = [genre,""]
    return ret

df_genres['Genres_tab_2'] = df_genres['Genres'].fillna('').apply(extract_2_genres)
print('Maximum legnth of tab = %d'  % df_genres['Genres_tab_2'].apply(len).max())
df_genres[['Genres_1','Genres_2']] = pd.DataFrame(df_genres.Genres_tab_2.values.tolist())
df_genres.head()
df_genres = df_genres[['App','Genres_1','Genres_2']]
df_genres_melted= pd.melt(df_genres, id_vars=['App'], value_vars=['Genres_1',"Genres_2"])
df_genres_melted['nb'] = 1
df_genres_melted.head()
df_genre_table = pd.pivot_table(df_genres_melted, values='nb', index=['App'],columns=['value'], aggfunc=np.sum).fillna(0).reset_index()
df_genre_table.drop(['','App'],axis=1,inplace=True)
arr_genre_sum = df_genre_table.sum()
df_genre_sum = pd.DataFrame(arr_genre_sum.sort_values(ascending=True)).reset_index()
df_genre_sum.columns = ['Genre','count']
df_genre_sum.head()
def plot_categorial(feat,figsize):
    sns.set(rc={'figure.figsize':figsize})
    return sns.countplot(y=feat, data=df,order=pd.DataFrame(df[feat].value_counts()).reset_index()['index'])

def describe_me(dataframe,key):
    print('The mean of %s is %f' % (key,dataframe[[key]].mean()))
    print('The median of %s is %f' % (key,dataframe[[key]].median()))
    print('The min of %s is %f' % (key,dataframe[[key]].min()))
    print('The max of %s is %f' % (key,dataframe[[key]].max()))
df_genre_sum.set_index('Genre').plot(kind='barh')
plot_categorial("Category",(5,8.27))
df['Category'].value_counts(normalize=True).cumsum()
plot_categorial("Type",(5,1))
plot_categorial("Content Rating",(5,3))
key = 'Rating'
describe_me(df,key)
df[[key]].hist(bins=25,figsize=(5,3))
key = 'Reviews'
describe_me(df,key)
df[[key]].hist(bins=10,figsize=(5,3),log=True)
reviews_noextrems = df[df.Reviews<100] 
describe_me(reviews_noextrems,key)
reviews_noextrems[[key]].hist(bins=30,figsize=(5,3))
key = 'Size'
describe_me(df,key)
df[[key]].hist(bins=25,figsize=(5,3))
key = 'Price'
charged = df[df.Price>0] 
describe_me(charged,key)
charged[[key]].hist(bins=25,figsize=(5,3),log=True)
charged_noextrems = df[np.logical_and(df.Price>0,df.Price<15)] 
describe_me(charged_noextrems,key)
charged_noextrems[[key]].hist(bins=30,figsize=(5,3))
key = 'since_last_update'
describe_me(df,key)
df[[key]].hist(bins=25,figsize=(5,3))

