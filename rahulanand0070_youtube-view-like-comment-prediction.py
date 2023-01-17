#Loading library

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import time

import warnings

warnings.filterwarnings('ignore')

import numpy as np

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier

from nltk.stem import PorterStemmer

import nltk

from sklearn.metrics import r2_score

youtube = pd.read_csv("../input/youtube-new/INvideos.csv")

youtube.head()
print(youtube.shape)
print(youtube.isnull().values.any())
youtube = youtube.dropna(how='any',axis=0)
youtube.describe()
youtube.drop(['video_id','thumbnail_link'],axis=1,inplace=True)
youtube.apply(lambda x: len(x.unique()))
for x in (['comments_disabled','ratings_disabled','video_error_or_removed','category_id']):

    count=youtube[x].value_counts()

    print(count)

    plt.figure(figsize=(7,7))

    sns.barplot(count.index, count.values, alpha=0.8)

    plt.title('{} vs No of video'.format(x))

    plt.ylabel('No of video')

    plt.xlabel('{}'.format(x))

    plt.show()
#No of tags

tags=[x.count("|")+1 for x in youtube["tags"]]

youtube["No_tags"]=tags
#length of desription

desc_len=[len(x) for x in youtube["description"]]

youtube["desc_len"]=desc_len
#length of title

title_len=[len(x) for x in youtube["title"]]

youtube["len_title"]=title_len
publish_time = pd.to_datetime(youtube['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

youtube['publish_time'] = publish_time.dt.time

youtube['publish_date'] = publish_time.dt.date



#day at which video is publish

youtube['publish_weekday']=publish_time.dt.weekday_name
#ratio of view/likes  upto 3 decimal

youtube["Ratio_View_likes"]=round(youtube["views"]/youtube["likes"],3)

#ratio of view/dislikes  upto 3 decimal

youtube["Ratio_View_dislikes"]=round(youtube["views"]/youtube["dislikes"],3)

#ratio of view/comment_count  upto 3 decimal

youtube["Ratio_views_comment_count"]=round(youtube["views"]/youtube["comment_count"],3)

#ratio of likes/dislikes  upto 3 decimal

youtube["Ratio_likes_dislikes"]=round(youtube["likes"]/youtube["dislikes"],3)
print(max(youtube["Ratio_View_likes"]))

print(max(youtube["Ratio_View_dislikes"]))

print(max(youtube["Ratio_views_comment_count"]))

print(max(youtube["Ratio_likes_dislikes"]))
#removing the infinite values

youtube=youtube.replace([np.inf, -np.inf], np.nan)

youtube = youtube.dropna(how='any',axis=0)
youtube['publish_weekday'] = youtube['publish_weekday'].replace({'Monday':1,

                                                             'Tuesday':2,

                                                             'Wednesday':3,

                                                             'Thursday':4,

                                                             'Friday':5,

                                                             'Saturday':6,

                                                             'Sunday':7})
count=youtube["publish_weekday"].value_counts()

print(count)

plt.figure(figsize=(7,7))

sns.barplot(count.index, count.values, alpha=0.8)

plt.title('No of videos vs weekdays')

plt.ylabel('no of videos')

plt.xlabel('weekdays')

plt.show()
data = youtube



corr = data.corr()

plt.figure(figsize=(12, 12))

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
youtube.drop(['trending_date','publish_date','publish_time','tags','title','description','channel_title'],axis=1,inplace=True)
views=youtube['views']

youtube_view=youtube.drop(['views'],axis=1,inplace=False)
train,test,y_train,y_test=train_test_split(youtube_view,views, test_size=0.2,shuffle=False)
print(train.shape,test.shape,y_train.shape,y_test.shape)

# REGRESSION ANALYSIS



# LINEAR REGRESSION



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



model = LinearRegression()

model.fit(train, y_train)



# predicting the  test set results

y_pred = model.predict(test)

print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

print("Result :",model.score(test, y_test))

d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}

SK = pd.DataFrame(data = d1)

print(SK)
lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)

fig1 = lm1.fig 

fig1.suptitle("Sklearn ", fontsize=18)

sns.set(font_scale = 1.5)
from sklearn.ensemble import RandomForestRegressor

nEstimator = [140,160,180,200,220]

depth = [10,15,20,25,30]



RF = RandomForestRegressor()

hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]

gsv = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)

gsv.fit(train, y_train)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))

plt.figure(figsize=(8, 8))

plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)

plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)

plt.xlabel('n_estimators')

plt.ylabel('max_depth')

plt.colorbar()

plt.xticks(np.arange(len(nEstimator)), nEstimator)

plt.yticks(np.arange(len(depth)), depth)

plt.title('Grid Search r^2 Score')

plt.show()

maxDepth=gsv.best_params_['max_depth']

nEstimators=gsv.best_params_['n_estimators']
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators = nEstimators,max_depth=maxDepth)

model.fit(train, y_train)





# predicting the  test set results

y_pred = model.predict(test)

print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

print("Result :",model.score(test, y_test))

d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}

SK = pd.DataFrame(data = d1)

print(SK)
lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)

fig1 = lm1.fig 

fig1.suptitle("Sklearn ", fontsize=18)

sns.set(font_scale = 1.5)

likes=youtube['likes']

youtube_like=youtube.drop(['likes'],axis=1,inplace=False)
train,test,y_train,y_test=train_test_split(youtube_like,likes, test_size=0.2,shuffle=False)
print(train.shape,test.shape,y_train.shape,y_test.shape)
# REGRESSION ANALYSIS



# LINEAR REGRESSION



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



model = LinearRegression()

model.fit(train, y_train)



# predicting the  test set results

y_pred = model.predict(test)

print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

print("Result :",model.score(test, y_test))

d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}

SK = pd.DataFrame(data = d1)

print(SK)
lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)

fig1 = lm1.fig 

fig1.suptitle("Sklearn ", fontsize=18)

sns.set(font_scale = 1.5)
from sklearn.ensemble import RandomForestRegressor



nEstimator = [140,160,180,200,220]

depth = [10,15,20,25,30]



RF = RandomForestRegressor()

hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]

gsv = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)

gsv.fit(train, y_train)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))



plt.figure(figsize=(8, 8))

plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)

plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)

plt.xlabel('n_estimators')

plt.ylabel('max_depth')

plt.colorbar()

plt.xticks(np.arange(len(nEstimator)), nEstimator)

plt.yticks(np.arange(len(depth)), depth)

plt.title('Grid Search r^2 Score')

plt.show()

maxDepth=gsv.best_params_['max_depth']

nEstimators=gsv.best_params_['n_estimators']
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators = nEstimators,max_depth=maxDepth)

model.fit(train, y_train)





# predicting the  test set results

y_pred = model.predict(test)

print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

print("Result :",model.score(test, y_test))

d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}

SK = pd.DataFrame(data = d1)

print(SK)
lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)

fig1 = lm1.fig 

fig1.suptitle("Sklearn ", fontsize=18)

sns.set(font_scale = 1.5)
comment_count=youtube['comment_count']

youtube_comment=youtube.drop(['comment_count'],axis=1,inplace=False)
train,test,y_train,y_test=train_test_split(youtube_comment,comment_count, test_size=0.2,shuffle=False)
print(train.shape,test.shape,y_train.shape,y_test.shape)

# REGRESSION ANALYSIS



# LINEAR REGRESSION



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



model = LinearRegression()

model.fit(train, y_train)



# predicting the  test set results

y_pred = model.predict(test)

print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

print("Result :",model.score(test, y_test))

d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}

SK = pd.DataFrame(data = d1)

print(SK)
lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)

fig1 = lm1.fig 

fig1.suptitle("Sklearn ", fontsize=18)

sns.set(font_scale = 1.5)
nEstimator = [140,160,180,200,220]

depth = [10,15,20,25,30]



RF = RandomForestRegressor()

hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]

gsv = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)

gsv.fit(train, y_train)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))



plt.figure(figsize=(8, 8))

plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)

plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)

plt.xlabel('n_estimators')

plt.ylabel('max_depth')

plt.colorbar()

plt.xticks(np.arange(len(nEstimator)), nEstimator)

plt.yticks(np.arange(len(depth)), depth)

plt.title('Grid Search r^2 Score')

plt.show()

maxDepth=gsv.best_params_['max_depth']

nEstimators=gsv.best_params_['n_estimators']
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators = nEstimators,max_depth=maxDepth)

model.fit(train, y_train)





# predicting the  test set results

y_pred = model.predict(test)

print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

print("Result :",model.score(test, y_test))

d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}

SK = pd.DataFrame(data = d1)

print(SK)
lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)

fig1 = lm1.fig 

fig1.suptitle("Sklearn ", fontsize=18)

sns.set(font_scale = 1.5)
