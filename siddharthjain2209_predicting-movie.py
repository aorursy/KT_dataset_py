import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\dell\\Desktop\\Sem_2\\Practical Data Science\\Project\\movie_metadata.csv',sep=',',decimal='.',header=0,encoding = "ISO-8859-1")
data.describe()
data.head(10)
data.info()
# Updating Missing Values



data['color'].fillna('Color',inplace=True)

data['num_critic_for_reviews'].fillna(0.0,inplace=True)

data['duration'].fillna(data['duration'].median(axis=0), inplace=True)

data['director_facebook_likes'].fillna(0.0,inplace=True)

data['actor_3_facebook_likes'].fillna(0.0,inplace=True)

data['actor_1_facebook_likes'].fillna(0.0,inplace=True)

data['gross'].fillna(data['gross'].median(),inplace=True)

data['facenumber_in_poster'].fillna(data['facenumber_in_poster'].median(),inplace=True)

data['num_user_for_reviews'].fillna(0.0,inplace=True)

data['budget'].fillna(data['budget'].median(),inplace=True)

data['actor_2_facebook_likes'].fillna(0.0,inplace=True)

data['language'].fillna('English',inplace=True)

data['country'].fillna('USA',inplace=True)

data['title_year'].fillna(2015,inplace=True)

data['aspect_ratio'].fillna(data['aspect_ratio'].median(),inplace=True)

data['movie_facebook_likes'].fillna(0.0,inplace=True)



data.head(10)
#Checking Values

data['color'].value_counts()



#Removing the blank spaces



data['color']=data['color'].str.strip()

data['color'].value_counts()
#Changing Data Type



data.imdb_score = data.imdb_score.astype(int)



data['num_critic_for_reviews']=data['num_critic_for_reviews'].astype(int)

data['duration']=data['duration'].astype(int)

data['director_facebook_likes']=data['director_facebook_likes'].astype(int)

data['actor_3_facebook_likes']=data['actor_3_facebook_likes'].astype(int)

data['actor_1_facebook_likes']=data['actor_1_facebook_likes'].astype(int)

data['gross']=data['gross'].astype(int)

data['facenumber_in_poster']=data['facenumber_in_poster'].astype(int)

data['num_user_for_reviews']=data['num_user_for_reviews'].astype(int)

data['budget']=data['budget'].astype(int)

data['actor_2_facebook_likes']=data['actor_2_facebook_likes'].astype(int)

data['title_year']=data['title_year'].astype(int)





data.head(5)

data.info()


(data_movie['country'].value_counts()).head(5).plot(kind='pie', title='Country', autopct='%.2f%%',fontsize=15, figsize=(6, 6))
fig, axes = plt.subplots(figsize=(15, 7))

(data['country'].value_counts()).head(10).plot(kind='bar')

plt.title('Bar Chart of Country')

plt.ylabel('Frequency')

plt.show()
(data_movie['language'].value_counts()).head(5).plot(kind='pie', title='Language spoken',autopct='%.2f%%')
data_movie['color'].value_counts().plot(kind='pie',title='Percentage of movies that are in colour',

autopct='%.2f%%',fontsize=20, figsize=(6, 6))

plt.show()
(data_movie['content_rating'].value_counts()).head(5).plot(kind='pie',title='Movie Classification',

autopct='%.2f%%',fontsize=15, figsize=(6, 6))

plt.show()
fig, axes = plt.subplots(figsize=(20, 7))

x=data_movie.title_year.value_counts().plot(kind='bar')

fig, axes = plt.subplots(figsize=(10, 7))

data_movie['actor_2_facebook_likes'].hist( bins=50); plt.title('Number of likes on Facebook for Actor 2'); 

plt.xlabel('Number of Likes'); 

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['actor_2_facebook_likes'].hist( bins=50); plt.title('Number of likes on Facebook for Actor 2 (log)'); 

plt.xlabel('Number of Likes'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['actor_3_facebook_likes'].hist( bins=50); plt.title('Number of likes on Facebook for Actor 3'); 

plt.xlabel('Number of Likes'); 

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['actor_3_facebook_likes'].hist( bins=50); plt.title('Number of likes on Facebook for Actor 3 (log)'); 

plt.xlabel('Number of Likes'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['actor_1_facebook_likes'].hist( bins=50); plt.title('Number of likes on Facebook for Actor 1'); 

plt.xlabel('Number of Likes'); 

plt.ylabel('Frequency')

plt.show()


fig, axes = plt.subplots(figsize=(10, 7))

data_movie['actor_1_facebook_likes'].hist( bins=50); plt.title('Number of likes on Facebook for Actor 1 (log)'); 

plt.xlabel('Number of Likes'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['num_critic_for_reviews'].hist( bins=10); plt.title('Number of reviews from critics ');

plt.xlabel('Number of reviews'); plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['num_user_for_reviews'].hist( bins=50); plt.title('Number of reviews from users (log)'); 

plt.xlabel('Number of Likes'); 

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['num_user_for_reviews'].hist( bins=50); plt.title('Number of reviews from users (log)'); 

plt.xlabel('Number of Likes'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['num_critic_for_reviews'].hist( bins=10); plt.title('Number of reviews from critics (log)');

plt.xlabel('Number of reviews'); plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['duration'].hist( bins=20); plt.title('Movie Duration'); plt.xlabel('Number of Minutes'); 

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['duration'].hist( bins=20); plt.title('Movie Duration'); plt.xlabel('Number of Minutes'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['duration'].plot(kind='box',title='Movie Duration')

plt.show()
fig, axes = plt.subplots(figsize=(7, 5))

plt.violinplot(data_movie['duration'],positions=None, vert=True, widths=0.5, showmeans=False, showextrema=True, showmedians=False, points=100, bw_method=None, hold=None, data=None)

plt.title('Violin Plot of Duration')

plt.ylabel('Frequency')
fig, axes = plt.subplots(figsize=(10, 5))

data_movie['duration'].plot(kind='box',title='Movie Duration')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['director_facebook_likes'].hist( bins=50); plt.title('Number of likes on Facebook for Director'); 

plt.xlabel('Number of Likes'); 

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['director_facebook_likes'].hist( bins=50); plt.title('Number of likes on Facebook for Director'); 

plt.xlabel('Number of Likes'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['cast_total_facebook_likes'].hist( bins=50); plt.title('Total Facebook likes of the cast'); plt.xlabel('Likes'); 

plt.ylabel('Frequency')

plt.show()

fig, axes = plt.subplots(figsize=(10, 7))

data_movie['cast_total_facebook_likes'].hist( bins=50); plt.title('Total Facebook likes of the cast (log)'); plt.xlabel('Likes'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['gross'].hist( bins=50); plt.title('Gross'); 

plt.xlabel('Dollars'); 

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['gross'].hist( bins=50); plt.title('Gross'); 

plt.xlabel('Dollars'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
data_movie['gross'].plot(kind='box',title='Movie Gross')

plt.show()
data_movie['budget'].plot(kind='box',title='Movie Budget')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['budget'].hist( bins=20); plt.title('Movie Budget'); 

plt.xlabel('Dollars'); 

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['budget'].hist( bins=50); plt.title('Movie Budget (log)'); 

plt.xlabel('Dollars'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['num_voted_users'].hist( bins=50); plt.title('Number of users voted'); plt.xlabel('Number of users'); 

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['num_voted_users'].hist( bins=50); plt.title('Number of users voted'); plt.xlabel('Number of users'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['facenumber_in_poster'].hist( bins=40); plt.title('Number of faces on poster'); plt.xlabel('Faces'); 

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['facenumber_in_poster'].hist( bins=40); plt.title('Number of faces on poster (log)'); plt.xlabel('Faces'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['facenumber_in_poster'].plot(kind='box',title='Number of faces on poster')

plt.show()
data_movie['facenumber_in_poster'].plot(kind='box',title='Number of faces on poster')

plt.yscale('log')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['movie_facebook_likes'].hist( bins=40); plt.title('Number of likes on Facebook for a movie'); 

plt.xlabel('Likes'); 

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(figsize=(10, 7))

data_movie['movie_facebook_likes'].hist( bins=40); plt.title('Number of likes on Facebook for a movie (log)'); 

plt.xlabel('Likes'); 

plt.ylabel('Frequency')

plt.yscale('log')

plt.show()


data_movie.plot(kind='scatter', x=25,y=15, title='Number of faces on poster vs. IMDB score', alpha=1, s=5)

plt.show()
data_movie.plot(kind='scatter', x=2,y=18, title='Critics review vs. Users review', alpha=0.75, s=1)

plt.show()


fig, axes = plt.subplots(figsize=(10, 7))

data_movie['imdb_score'].hist( bins=7); plt.title('IMDB score'); 

plt.xlabel('Rating from 0 (bad) to 10 (great)'); 

plt.ylabel('Frequency')

plt.show()
data_movie.plot(kind='scatter', x=25,y=12, title='IMDB score vs. Number of votes by users', alpha=0.75, s=5)

plt.show()
data_movie.plot(kind='scatter', x=25,y=13, title='IMDB score vs Total cast Facebook likes', alpha=0.75, s=5)

plt.yscale('log')

plt.ylim((1,1000000))

plt.show()
data_movie.plot(kind='scatter', x=25,y=22, title='IMDB score vs. Movie Budget', alpha=0.75, s=5)

plt.yscale('log')

plt.ylim((100,10000000000))

plt.show()
data_movie.plot(kind='scatter', x=25,y=22, title='IMDB score vs. Movie Budget', alpha=0.75, s=5)

plt.yscale('log')

plt.show()
data_movie.plot(kind='scatter', x=25,y=8, title='IMDB score vs. Movie Takings', alpha=0.75, s=5)

plt.show()
data_movie.plot(kind='scatter', x=25,y=27, title='IMDB score vs. Facebook likes for movie', alpha=0.75, s=5)

plt.yscale('log')

plt.ylim((1,1000000))

plt.show()
data_movie.boxplot(column='imdb_score',by='title_year',figsize=(20,7))
data.boxplot(column='movie_facebook_likes',by='IQR')

plt.show()
xs=data['IQR']

ys=data['num_voted_users']

zs=data['gross']



import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs,ys,zs,zdir='z', s=40, c=None, depthshade=True)
plt.scatter(data['IQR'],data['num_voted_users'], label='skitscat', color='k', s=25, marker="o")
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
sns.pairplot(data)
xs=data['IQR']

ys=data['num_user_for_reviews']

zs=data['gross']



import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.bar(xs,ys, zs, zdir='z')

names = ["num_critic_for_reviews","duration","director_facebook_likes","actor_3_facebook_likes","actor_1_facebook_likes",

        "gross","num_voted_users","cast_total_facebook_likes","facenumber_in_poster","num_user_for_reviews","budget",

        "title_year","actor_2_facebook_likes","movie_facebook_likes","imdb_score"]

data = data_iqr

correlations = data.corr()

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1,cmap=plt.cm.rainbow)

fig.colorbar(cax, ax=ax)

ticks = np.arange(0,15,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names, rotation=90)

ax.set_yticklabels(names)



plt.show()



iqr = pd.qcut(data_movie["imdb_score"], 4) # Number of clusters - we have created 4 (bad, average, good and great)

iqr = pd.qcut(data_movie["imdb_score"], 4, labels=range(4))

data.drop(data.columns[[25]], axis=1, inplace=True) # remove the float IMDB score 

data_iqr = pd.concat([data, iqr], axis=1) # replaced the float with the IQR or cluster

data_iqr.imdb_score = data_iqr.imdb_score.astype(int)
data_iqr=data
data_iqr.info()
data_iqr.drop(data_iqr.columns[[0,1,6,9,10,11,14,16,17,19,20,21,25,28]], axis=1, inplace=True)
data_iqr.info()

data_iqr.head(5)
data_iqr['IQR_NEW'].hist( bins=4); plt.title('IMDB score'); 

plt.xlabel('Rating from 0 (bad) to 10 (great)'); 

plt.ylabel('Frequency')

plt.show()
data_iqr['IQR_NEW'].hist( bins=7); plt.title('IMDB score'); 

plt.xlabel('Rating from 0 (bad) to 10 (great)'); 

plt.ylabel('Frequency')

plt.show()
X = data_iqr.drop('IQR_NEW', axis=1)

y = data_iqr['IQR_NEW']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

data_iqr.info()


from sklearn import preprocessing

from sklearn.cross_validation import train_test_split

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn import tree

from math import sqrt
keys = []

scores = []



models = {'K-Nearest Neighbors':KNeighborsClassifier(n_neighbors=20, weights='uniform',

            algorithm='auto', p=2, metric='minkowski', metric_params=None), 

          'Decision Tree': DecisionTreeClassifier(criterion='gini', splitter='best', 

            max_depth=4, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 

            max_features=15, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, 

            class_weight=None, presort=False),

          'Random Forest': RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, 

            min_samples_split=50, min_samples_leaf=20, min_weight_fraction_leaf=0.0, max_features=4, 

            max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=3, 

            random_state=None, verbose=0, warm_start=True, class_weight=None)}

for k,v in models.items():

    mod = v

    mod.fit(X_train, y_train)

    pred = mod.predict(X_test)

    print('Results for: ' + str(k) + '\n')

    print(confusion_matrix(y_test, pred))

    print(classification_report(y_test, pred))

    acc = accuracy_score(y_test, pred)

    print(acc)

    print('\n' + '\n')

    keys.append(k)

    scores.append(acc)

    table = pd.DataFrame({'model':keys, 'accuracy score':scores})



print(table)
keys = []

scores = []

models = {'Decision Tree': DecisionTreeClassifier(criterion='gini', splitter='best',

            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 

            max_features=4, max_leaf_nodes=None,class_weight=None, presort=False)}

for k,v in models.items():

    mod = v

    mod.fit(X_train, y_train)

    pred = mod.predict(X_test)

    print('Results for: ' + str(k) + '\n')

    print(confusion_matrix(y_test, pred))

    print(classification_report(y_test, pred))

    acc = accuracy_score(y_test, pred)

    print(acc)

    print('\n' + '\n')

    keys.append(k)

    scores.append(acc)

    table = pd.DataFrame({'model':keys, 'accuracy score':scores})



print(table)





clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

tree.export_graphviz(mod, out_file='tree.dot')


# calculate the fpr and tpr for all thresholds of the classification

probs = model.predict_proba(X_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



data_iqr['IQR_NEW'].value_counts()
rfc = RandomForestClassifier(n_estimators = 14)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)



importances = rfc.feature_importances_

plot = sns.barplot(x=X.columns, y=importances)



for item in plot.get_xticklabels():

   item.set_rotation(90)
def f(data):

    if (data['imdb_score'] in (1,2,3)):

        val = 0

    elif (data['imdb_score'] in (4,5)):

        val = 1

    elif (data['imdb_score'] in (6,7)):

        val = 2

    else:

        val = 3

    return val
data['IQR_NEW'] = data.apply(f, axis=1)
data.head(10)
data['IQR_NEW'].value_counts()
data.drop('IQR_NEW',axis=1)