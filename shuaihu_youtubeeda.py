import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

import os

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS
df_yout = pd.read_csv("../input/youtube-new/USvideos.csv")
print(df_yout.shape)

print(df_yout.nunique())
df_yout.head(n=2)
df_yout = df_yout.drop_duplicates(subset=['video_id','trending_date'], keep='last', inplace=False) #duble drop



df_yout['trending_times'] = np.nan #compute trending_times

for v_id in df_yout['video_id'].unique():

    trending_times = sum(df_yout['video_id'] == v_id)

    df_yout.loc[(df_yout["video_id"] == v_id),"trending_times"] = trending_times



df_youtube = df_yout.drop_duplicates(subset='video_id', keep='last', inplace=False) #drop
df_yout = df_yout.drop_duplicates(subset='video_id', keep='last', inplace=False) #drop
df_yout.info
df_yout['category_name'] = np.nan



df_yout.loc[(df_yout["category_id"] == 1),"category_name"] = 'Film and Animation'

df_yout.loc[(df_yout["category_id"] == 2),"category_name"] = 'Cars and Vehicles'

df_yout.loc[(df_yout["category_id"] == 10),"category_name"] = 'Music'

df_yout.loc[(df_yout["category_id"] == 15),"category_name"] = 'Pets and Animals'

df_yout.loc[(df_yout["category_id"] == 17),"category_name"] = 'Sport'

df_yout.loc[(df_yout["category_id"] == 19),"category_name"] = 'Travel and Events'

df_yout.loc[(df_yout["category_id"] == 20),"category_name"] = 'Gaming'

df_yout.loc[(df_yout["category_id"] == 22),"category_name"] = 'People and Blogs'

df_yout.loc[(df_yout["category_id"] == 23),"category_name"] = 'Comedy'

df_yout.loc[(df_yout["category_id"] == 24),"category_name"] = 'Entertainment'

df_yout.loc[(df_yout["category_id"] == 25),"category_name"] = 'News and Politics'

df_yout.loc[(df_yout["category_id"] == 26),"category_name"] = 'How to and Style'

df_yout.loc[(df_yout["category_id"] == 27),"category_name"] = 'Education'

df_yout.loc[(df_yout["category_id"] == 28),"category_name"] = 'Science and Technology'

df_yout.loc[(df_yout["category_id"] == 29),"category_name"] = 'Non Profits and Activism'

df_yout.loc[(df_yout["category_id"] == 25),"category_name"] = 'News & Politics'
df_yout['likes_log'] = np.log(df_yout['likes'] + 1)

df_yout['views_log'] = np.log(df_yout['views'] + 1)

df_yout['dislikes_log'] = np.log(df_yout['dislikes'] + 1)

df_yout['comment_log'] = np.log(df_yout['comment_count'] + 1)
plt.figure(figsize = (14,9))



g = sns.boxplot(x='category_name', y='views_log', data=df_yout, palette="Set3")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Views Distribuition by Category Names", fontsize=20)

g.set_xlabel("", fontsize=15)

g.set_ylabel("Views(log)", fontsize=15)



plt.show()
df_yout['publish_date']=df_yout['publish_time'].apply(lambda x : x.split('T')[0])
df_yout['weekday']=pd.to_datetime(df_yout['publish_date']).dt.weekday_name

df_yout['weekday']
plt.figure(figsize = (14,9))



g = sns.boxplot(x='weekday', y='views_log', data=df_yout, palette="Set3")

g.set_xticklabels(g.get_xticklabels())

g.set_title("Views Distribuition by Publish Date", fontsize=20)

g.set_xlabel("", fontsize=15)

g.set_ylabel("Views(log)", fontsize=15)



plt.show()
df_yout['weekday']=pd.to_datetime(df_yout['publish_date']).dt.weekday
df_yout['publish_time']=df_yout['publish_time'].apply(lambda x : x.split('T')[1])
df_yout['publish_hour']=df_yout['publish_time'].apply(lambda x : int(x[0:2]))
plt.figure(figsize = (14,9))



g = sns.boxplot(x='publish_hour', y='views_log', data=df_yout, palette="Set3")

g.set_xticklabels(g.get_xticklabels())

g.set_title("Views Distribuition by Publish time", fontsize=20)

g.set_xlabel("Publish time(hour)", fontsize=15)

g.set_ylabel("Views(log)", fontsize=15)



plt.show()
df_yout['description'].unique()
df_yout['title_len']=df_yout['title'].apply(lambda x : len(x))

df_yout['description_len']=df_yout['description'].apply(lambda  x: len(str(x)))

df_yout['tags_number']=df_yout['tags'].apply(lambda x : len(x.split("|")))
# df_yout[['title_len','description_len','tags_number']].nunique()

plt.figure(figsize = (12,10))

plt.title('Length of Title vs. Number of Views', size= 30)

plt.xlabel('length of tiltle')

plt.ylabel('number of views(log)')

plt.scatter(df_yout['title_len'], df_yout['views_log'], marker='o', alpha = .3)

plt.show()
plt.figure(figsize = (12,10))

plt.title('Length of Description vs. Number of Views', size= 30)

plt.xlabel('length of description')

plt.ylabel('number of views(log)')

plt.scatter(df_yout['description_len'], df_yout['views_log'], marker='o', alpha = .3)

plt.show()
plt.figure(figsize = (12,10))

plt.title('Number of Tags vs. Number of Views', size= 30)

plt.xlabel('number of tags')

plt.ylabel('number of views(log)')

plt.scatter(df_yout['tags_number'], df_yout['views_log'], marker='o', alpha = .3)

plt.show()
#  df_yout[['views_log','likes_log','dislikes_log','comment_log','trending_times','publish_hour','weekday','title_len','category_id','tags_number','description_len']]
pearsoncorr = df_yout[['views_log','likes_log','dislikes_log','comment_log','trending_times','publish_hour','weekday','title_len','category_id','tags_number','description_len']].corr(method='pearson')
# pearsoncorr = df_yout[['views','likes','dislikes','comment_count','trending_times']].corr(method='pearson')

# pearsoncorr
plt.figure(figsize = (18,15))

plt.title('Pearson Correlation of Features', size= 30)

sns.set(font_scale=1.8) 

sns.heatmap(pearsoncorr,linewidths=0.1,vmax = 1, square = True,cmap = plt.cm.RdBu, linecolor='white',annot_kws={'size':15}, annot=True,)

plt.show()

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

data_X = df_yout[['publish_hour','weekday','category_id']]

data_y = df_yout['trending_times']

# data_X = np.array(data_X)

# data_X = data_X.reshape(-1,1)

feat_enc = OneHotEncoder()

feat_enc.fit(data_X)

data_X = feat_enc.transform(data_X)

data_X = pd.DataFrame(data_X.todense())



tags = df_yout['tags']

n = len(tags)

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words="english",analyzer='word', max_df=1.0, min_df=50/n+0.0001, max_features=None)

tags_vectors = count_vectorizer.fit_transform(tags)

tags_vectors = pd.DataFrame(tags_vectors.todense())

# test_vectors = count_vectorizer.transform(X_test)

# print(len(count_vectorizer.get_feature_names()))

# print(count_vectorizer.get_feature_names())



# tags_vectors = np.array(tags_vectors)

# tags_vectors.todense()

data_X = pd.concat([data_X, tags_vectors],axis = 1, sort=False)

description_len =  df_yout['description_len'].reset_index(drop=True) 

description_len

data_X = pd.concat([data_X, description_len],axis = 1, sort=False)

data_X



X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, train_size=0.9)



X_train
np.arange(1,len(df_yout['description_len']))
# tags = X_train['tags']

# tags_test = X_test['tags']
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# count_vec = CountVectorizer(stop_words="english", analyzer='word', ngram_range=(1, 1), max_df=0.50, min_df=1, max_features=None)



#2 times

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words="english",analyzer='word', max_df=1.0, min_df=1/n+0.0001, max_features=None)

train_vectors = count_vectorizer.fit_transform(tags)

test_vectors = count_vectorizer.transform(tags)

print(len(count_vectorizer.get_feature_names()))

print(count_vectorizer.get_feature_names())

# train_vectors = train_vectors.toarray()

# test_vectors = test_vectors.toarray()
#10 times

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words="english",analyzer='word', max_df=1.0, min_df=10/n+0.0001, max_features=None)

train_vectors = count_vectorizer.fit_transform(tags)

test_vectors = count_vectorizer.transform(tags)

print(len(count_vectorizer.get_feature_names()))

print(count_vectorizer.get_feature_names())
#50 times

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words="english",analyzer='word', max_df=1.0, min_df=50/n+0.0001, max_features=None)

train_vectors = count_vectorizer.fit_transform(tags)

test_vectors = count_vectorizer.transform(tags)

print(len(count_vectorizer.get_feature_names()))

print(count_vectorizer.get_feature_names())
#100 times

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words="english",analyzer='word', max_df=1.0, min_df=100/n+0.0001, max_features=None)

train_vectors = count_vectorizer.fit_transform(tags)

test_vectors = count_vectorizer.transform(tags)

print(len(count_vectorizer.get_feature_names()))

print(count_vectorizer.get_feature_names())
#150 times

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words="english",analyzer='word', max_df=1.0, min_df=150/n+0.0001, max_features=None)

train_vectors = count_vectorizer.fit_transform(tags)

test_vectors = count_vectorizer.transform(tags)

print(len(count_vectorizer.get_feature_names()))

print(count_vectorizer.get_feature_names())
#200 times

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words="english",analyzer='word', max_df=1.0, min_df=200/n+0.0001, max_features=None)

train_vectors = count_vectorizer.fit_transform(tags)

test_vectors = count_vectorizer.transform(tags)

print(len(count_vectorizer.get_feature_names()))

print(count_vectorizer.get_feature_names())
#500 times

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words="english",analyzer='word', max_df=1.0, min_df=500/n+0.0001, max_features=None)

train_vectors = count_vectorizer.fit_transform(tags)

test_vectors = count_vectorizer.transform(tags)

print(len(count_vectorizer.get_feature_names()))

print(count_vectorizer.get_feature_names())
# X_train = X_train.to_numpy()

# X_train = X_train.reshape(-1,1)

# type(X_train)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

reg = LinearRegression()

# cross_val_score(reg, train_vectors, y_train, cv=3)

print(cross_val_score(reg, X_train, y_train, cv=3))

# result = reg.predict(test_vectors)

# print(mean_squared_error(result,y_test)**0.5/np.mean(y_test))

# print(reg.score(test_vectors, y_test))
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

reg = MLPRegressor()

# cross_val_score(reg, train_vectors, y_train, cv=3)

print(cross_val_score(reg, X_train, y_train, cv=3))
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor







# Fit regression model

regr_1 = DecisionTreeRegressor(max_depth=6)



regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),

                          n_estimators=50)



regr_1.fit(X_train, y_train)

regr_2.fit(X_train, y_train)



# Predict

# y_1 = regr_1.predict(X)

# y_2 = regr_2.predict(X)

print(regr_1.score(X_test, y_test))

print(regr_2.score(X_test, y_test))

print(cross_val_score(regr_1, X_train, y_train, cv=3))
#150 times

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words="english",analyzer='word', max_df=1.0, min_df=150/n+0.0001, max_features=None)

train_vectors = count_vectorizer.fit_transform(tags)

test_vectors = count_vectorizer.transform(tags)

print(len(count_vectorizer.get_feature_names()))

print(count_vectorizer.get_feature_names())
reg = LinearRegression()

reg.fit(train_vectors, data_y)

d = {'coef': reg.coef_, 'tag':count_vectorizer.get_feature_names()}

df = pd.DataFrame(data=d)

df.sort_values(by=['coef'])
# df_youtube = pd.read_csv("../input/youtube-new/USvideos.csv")

# print(df_youtube.shape)

# print(df_youtube.nunique())
# df_youtube = df_youtube.drop_duplicates(subset=['video_id','trending_date'], keep='last', inplace=False)

# print(df_youtube.shape)

# print(df_youtube.nunique())
# df_youtube['trending_times'] = np.nan

# n=0

# for v_id in df_youtube['video_id'].unique():

#     trending_times = sum(df_youtube['video_id'] == v_id)

#     n+=1

#     print(n)

#     df_youtube.loc[(df_youtube["video_id"] == v_id),"trending_times"] = trending_times

#     df_youtube.loc[df_youtube['video_id'] == v_id]["trending_times"]=trending_times

# df_youtube.loc[df_youtube['video_id'] == 'ooyjaVdt-jA']
# df_youtube = df_youtube.drop_duplicates(subset='video_id', keep='last', inplace=False)
# print(df_youtube.shape)

# print(df_youtube.nunique())