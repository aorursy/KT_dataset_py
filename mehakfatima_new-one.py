import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
from sklearn import preprocessing
import scipy.stats as stats
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
bolly = pd.read_csv('../input/boxoffh/movie_metadata.csv')
bolly.keys()
bolly.describe()
data=bolly.fillna(0)
data[['movie_title', 'num_voted_users', 'title_year']].sort_values('num_voted_users', ascending=False).head(10)
title_corpus = ' '.join(data['movie_title'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()
print (data.shape)
from pandas import Series, DataFrame
s = data['genres'].str.split('|').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'genres'
del data['genres']
df = data.join(s)
df.head()
df['genres'].unique()
len(df['genres'].unique())
df1 = df[df['imdb_score']>=7]
df1.head()
df2 = (pd.DataFrame(df1.groupby('genres').movie_title.nunique())).sort_values('movie_title', ascending=False )
df2
df2[['movie_title']].plot.barh(stacked=True, title = 'Genres with >= 7 ratings', figsize=(10, 8));
import pylab as pl

imdbScore=[[]]
x=[]

for i in pl.frange(1,9.5,.5):
    imdbScore.append(len(data.imdb_score[(data.imdb_score>=i) & (data.imdb_score<i+.5)]))
    x.append(i)

del(imdbScore[0])

plt.figure(figsize=(15,12))
plt.title("Histogram Of IMDB Score")
plt.ylabel("IMDB Score")
plt.xlabel('Frequency')
plt.barh(x,imdbScore,height=.45 ,color='green')
plt.yticks(x)
plt.show()
plt.figure(figsize=(12,10))
plt.title("IMDB Score Vs Director Facebook Popularity")
plt.xlabel("IMDB Score")
plt.ylabel("Facebook Popularity")
tmp=plt.scatter(data.imdb_score,data.director_facebook_likes,c=data.imdb_score,vmin=3,vmax=10)
plt.yticks([i*2500 for i in range(11)])
plt.colorbar(tmp,fraction=.025)
plt.show()
plt.figure(figsize=(12,10))
plt.title("IMDB Score Vs Cast Facebook Popularity")
plt.xlabel("IMDB Score")
plt.ylabel("Facebook Popularity")
tmp=plt.scatter(data.imdb_score,data.cast_total_facebook_likes,c=data.imdb_score,vmin=3,vmax=10)
plt.yticks([i*70000 for i in range(11)])
plt.colorbar(tmp,fraction=.025)
plt.show()
#bollyw = bollyw.select_dtypes(include=['int64','float64']) #only float and int values
#print (bollyw.shape)
corr = data.corr()
print (corr)
corr_mat=data.corr(method='pearson')
plt.figure(figsize=(20,10))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
features = ['duration','num_critic_for_reviews','num_user_for_reviews']
target = ['imdb_score']
#X = data[features]
#Y =data[target]
sns.pairplot(data, x_vars=['duration','num_critic_for_reviews','num_user_for_reviews'], y_vars='imdb_score', size=7, aspect=0.7)
from sklearn.model_selection import train_test_split
train, test = train_test_split(data,test_size=0.20)

X_train = train[features].dropna()
Y_train = train[target].dropna()
X_test = test[features].dropna()
Y_test = test[target].dropna()
X_test.head()
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_test,Y_test)
lin_score_train = lin.score(X_test, Y_test)
lin_score_test = lin.score(X_train, Y_train)
print(lin_score_train)
print(lin_score_test)
pred_train=lin.fit(X_train,Y_train)
#=lin.predict(X_train)
pred_test=lin.predict(X_test)
print (pred_train)
m=lin.coef_
b=lin.intercept_
print("slope=",m, "intercept=",b)

print ('coffecient:',len(lin.coef_))
from sklearn import metrics
print ('Mean abs error',metrics.mean_absolute_error(Y_test,pred_test))
print ('mean sq error',metrics.mean_squared_error(Y_test ,pred_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, pred_test)))
sns.pairplot(data, x_vars=['duration','num_user_for_reviews','num_critic_for_reviews'], y_vars='imdb_score', size=7, aspect=0.7, kind='reg')
data['highvoted'] = data['imdb_score'].map(lambda s :1  if s >= 7 else 0)
data.loc[:, ['imdb_score', 'highvoted']].head()
ind = ['num_voted_users']
dep = ['imdb_score']
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(ind, dep)
y_pred = regressor.predict(ind)
#from sklearn.model_selection import train_test_split
#train, test = train_test_split(data,test_size=0.20)

X_xtrain = train[features2]
Y_ytrain = train[target2]
X_xtest = test[features2]
Y_ytest = test[target2]
X_xtest.head()

#profit = (((bollyw['gross'].values)-(bollyw['budget'].values))/(bollyw['gross'].values))*100
#bollyw.loc[:,'profit'] = pd.Series(profit, index=bollyw.index)


print ({'Actual': Y_ytest, 'Predicted': y_pred})
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_ytest, pred_test))
print('Mean Squared Error:', metrics.mean_squared_error(Y_ytest, pred_test))
print('Root Mean Squared Error:',
np.sqrt(metrics.mean_squared_error(Y_ytest, pred_test)))
df = data.loc[:, (data != 0).any(axis=0)]
features1 = ['director_facebook_likes', 'actor_3_facebook_likes','gross','duration','num_critic_for_reviews','actor_1_facebook_likes', 'cast_total_facebook_likes', 'num_user_for_reviews', 'title_year', 'actor_2_facebook_likes','aspect_ratio', 'movie_facebook_likes']
target1 = ['highvoted']
sns.pairplot(data, x_vars=['num_critic_for_reviews','duration','num_user_for_reviews','movie_facebook_likes','imdb_score'], y_vars='gross', size=7, aspect=0.7)
from sklearn.model_selection import train_test_split
train, test = train_test_split(data,test_size=0.25)

x_train = train[features1].dropna()
y_train = train[target1].dropna()
x_test = test[features1].dropna()
y_test = test[target1].dropna()
x_test.head()

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train) 
y_pred = classifier.predict(x_test)
score=classifier.score(x_test,y_test)
print(score)
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.model_selection import  cross_val_score

c_dec = cross_val_score(classifier, x_train, y_train, cv=10)
c_dec.mean()
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True,
cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
plt.show()
data['country'].unique()
df_countries = df['title_year'].groupby(df['country']).count()
df_countries = df_countries.reset_index()
df_countries.rename(columns ={'title_year':'count'}, inplace = True)
df_countries = df_countries.sort_values('count', ascending = False)
df_countries.reset_index(drop=True, inplace = True)
sns.set_context("poster", font_scale=0.6)
plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(13, 8))
labels = [s[0] if s[1] > 80 else ' ' 
          for index, s in  df_countries[['country', 'count']].iterrows()]
sizes  = df_countries['count'].values
explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(df_countries))]
ax.pie(sizes, explode = explode, labels = labels,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow=False, startangle=45)
ax.axis('equal')
ax.set_title('Percentage of films per country',
             bbox={'facecolor':'k', 'pad':5},color='w', fontsize=16);
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
df = data
my_imputer = Imputer()
X2 = my_imputer.fit_transform(df[['duration']])
df['duration'] = X2

budget = ['budget','duration']
training = df[budget]

target= ['num_voted_users']
target = df[target]

X = training.values
y = target.values

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y, test_size=0.3)

# Create linear regression object
regr1 = linear_model.LinearRegression()

# Train the model using the training sets
regr1.fit(X_train1, y_train1)

# Make predictions using the testing set
y_pred_lr1 = regr1.predict(X_test1)

print(regr1.coef_, regr1.intercept_)
print(r2_score(y_test1, y_pred_lr1))

f = plt.figure(figsize=(10,5))
plt.scatter(X_test1[:,1], y_test1, label="Real score");
plt.scatter(X_test1[:,1], y_pred_lr1, c='r',label="Predicted score");
plt.xlabel("Budget");
plt.ylabel('Score');
plt.legend(loc=2);
from sklearn.ensemble import RandomForestRegressor
rf1 = RandomForestRegressor(1)
# Train the model using the training sets
rf1.fit(X_train1, y_train1)
# Make predictions using the testing set
y_pred_rf1 = rf1.predict(X_test1)
f = plt.figure(figsize=(10,5))
plt.scatter(X_test1[:,1], y_test1, s=50,label="Real Score");
plt.scatter(X_test1[:,1], y_pred_rf1,s=100, c='r',label="Predicted Score");
plt.ylabel("Score");
plt.legend(loc=2);
error_lr = mean_squared_error(y_test1,y_pred_lr1)
error_rf = mean_squared_error(y_test1,y_pred_rf1)
print(error_lr)
print(error_rf)

f = plt.figure(figsize=(10,5))
plt.bar(range(2),[error_lr,error_rf])
plt.xlabel("Classifiers");
plt.ylabel("Mean Squared Error of the Score");
plt.xticks(range(2),['Linear Regression','Random Forest'])
plt.legend(loc=2);
