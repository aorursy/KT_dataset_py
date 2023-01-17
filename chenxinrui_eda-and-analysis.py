import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import math

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

plt.style.use('classic')

%matplotlib inline
# read in datasets

app = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

review = pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')
# clean data (only for further analysis purposes, complete code and comments please refer to EDA)

app = app.drop_duplicates(subset=['App'], keep='first')

app = app[(app.Price != 'Everyone' )]

app.Price = app.dropna().Price.apply(lambda x: x.replace('$',''))

app['Price'] = app['Price'].astype(float)

app['Last Updated'] = pd.to_datetime(app['Last Updated'] )

app.Installs = app.dropna().Installs.apply(lambda x: x.replace('+','').replace(',',''))

app['Installs'] = app['Installs'].dropna().astype(int)

app['Reviews'] = app['Reviews'].astype(float)

app.loc[app.Type == 'Free', 'Price'] = 0

app.Rating = app.Rating.fillna(app.Rating.mean())

review = review.dropna(axis = 'rows') 
app1 = app[app['Price']<=40].dropna() # most apps are under 40 dollars, filter to make it easier to see the trend

plt.rcParams['figure.figsize'] = (8, 4)

ratings = app1['Rating']

prices = app1['Price'] 

ins = app1['Installs']

plt.scatter(ratings, prices, c= np.log(ins),alpha = 0.3,label = None) # installs is very large, so I log them

plt.xlabel('ratings')

plt.ylabel('prices')

plt.colorbar(label = 'log$_{10}$(installs)')

plt.clim(3,6)

plt.title('Ratings,installs and Prices ')

plt.show;
df = app.groupby(['Category'])[['Rating', 'Installs', 'Price']].mean()
# scale

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df)

df_scaled=scaler.transform(df)
from sklearn.decomposition import PCA

pca = PCA(n_components=2) 

pca.fit(df_scaled)

pca_loadings = pca.components_
plt.figure()

for i in range(2):

    plt.subplot(1,2,i+1)

    heights = np.squeeze(pca_loadings[i,:])

    bars = ['Rating', 'Installs', 'Price']

    x_pos = np.arange(len(bars))

    plt.bar(x_pos, heights)

    plt.xticks(x_pos, bars)

    plt.title("Loadings PC "+str(i+1))

    plt.ylim(-1,1)

print(i);
pca_scores = pca.fit_transform(df_scaled)

plt.rcParams['figure.figsize'] = (17, 8)

# Reference: https://github.com/teddyroland/python-biplot/blob/master/biplot.py

xvector = pca.components_[0] 

yvector = pca.components_[1]

xs = pca_scores[:,0]

ys = pca_scores[:,1]

for i in range(len(xvector)):

# arrows project features (ie columns from csv) as vectors onto PC axes

    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),

              color='r', width=0.0005, head_width=0.0025)

    plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,

             list(df.columns.values)[i], color='r')



for i in range(len(xs)):

# circles project documents (ie rows from csv) as points onto PC axes

    plt.plot(xs[i], ys[i], 'bo')

    plt.text(xs[i]*1.2, ys[i]*1.2, list(df.index)[i], color='b')

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component')

plt.show();
plt.rcParams['figure.figsize'] = (8, 4)

plt.scatter(x=np.log(app.Reviews), y=app.Rating,alpha=0.1)

plt.ylabel('ratings')

plt.xlabel('log(reviews)')

plt.title('reviews and rating');
plt.rcParams['figure.figsize'] = (8, 4)

plt.scatter(x=np.log(app.Installs), y=app.Rating,alpha=0.1)

plt.ylabel('ratings')

plt.xlabel('log(Installs)')

plt.title('Installs and rating');
plt.rcParams['figure.figsize'] = (8, 4)

plt.scatter(x=app.Price, y=app.Rating,alpha=0.1)

plt.ylabel('ratings')

plt.xlabel('prices')

plt.xlim(-1,40)

plt.title('price and rating');
corr=app[['Rating','Installs','Price','Reviews']].corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_yticklabels(

    ax.get_yticklabels(),

    rotation=0,

);
plt.rcParams['figure.figsize'] = (8, 4)

sns.boxplot(y = app['Rating'], x = app['Content Rating'],width=0.5, linewidth=0.8,fliersize=1)

plt.xticks(rotation=90)

plt.ylim(0,6)

plt.title('Ratings in Different Content Rating Categories',fontsize = 20)

plt.ylabel('Ratings',fontsize = 15)

plt.xlabel('Content Rating',fontsize = 15)

plt.show();
plt.rcParams['figure.figsize'] = (15, 6)

sns.boxplot(y = app['Rating'], x = app['Genres'],width=0.5, linewidth=0.8,fliersize=1)

plt.xticks(rotation=90)

plt.ylim(0,6)

plt.title('Ratings in Different Genres',fontsize = 20)

plt.ylabel('Ratings',fontsize = 15)

plt.xlabel('Genres',fontsize = 15)

plt.show();
plt.rcParams['figure.figsize'] = (15, 6)

sns.boxplot(y = app['Rating'], x = app['Category'],width=0.5, linewidth=0.8,fliersize=1)

plt.xticks(rotation=90)

plt.ylim(0,8)

plt.title('Ratings in Different Categories',fontsize = 20)

plt.ylabel('ratings',fontsize = 15)

plt.xlabel('category',fontsize = 15)

plt.show();
combine = pd.merge(app,review,on='App',how='left')
combine.head(5)
#calculate the proportion of neutral,positive,negative reviews

sentiment_count = combine.groupby(['App','Sentiment']).count()

proportion = sentiment_count['Category']

proportion = proportion.groupby(level=0).apply(lambda x:

                             x / float(x.sum()))
p = proportion.to_frame()

p = p.reset_index(level=['App', 'Sentiment']) 

# select neutral,positive,negative reviews separately

propor_po = p[p['Sentiment'].isin(['Positive'])][['App','Category']]

propor_ne = p[p['Sentiment'].isin(['Negative'])][['App','Category']]

propor_neu = p[p['Sentiment'].isin(['Neutral'])][['App','Category']]

# change column names

propor_po.columns = ['App', 'positive_proportion']

propor_ne.columns = ['App','negative_proportion']

propor_neu.columns = ['App','neutral_proportion']

# calculate average sentiment polarity and subjectivity

sentiment_polar = pd.DataFrame(combine.dropna().groupby(['App'])['Sentiment_Polarity'].mean()).reset_index()

sentiment_sub = pd.DataFrame(combine.dropna().groupby(['App'])['Sentiment_Subjectivity'].mean()).reset_index()

# merge all these data frames at the same time

from functools import reduce

data_frames = [app,propor_po, propor_ne, propor_neu,sentiment_polar,sentiment_sub]

sentiment3 = reduce(lambda  left,right: pd.merge(left,right,on=['App'],

                                            how='outer'), data_frames)

sentiment3.head(3)
features = ['Rating','Reviews','Installs','Price','positive_proportion','negative_proportion','neutral_proportion','Sentiment_Polarity','Sentiment_Subjectivity']

# Separating out the features

x = sentiment3[features].dropna()

corr = x.corr()

corr
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
scaler = StandardScaler()

scaler.fit(x)

x_scaled=scaler.transform(x)

pca = PCA() 

pca.fit(x_scaled)

print('The variance explained by the components is \n' + str(pca.explained_variance_ratio_))
plt.rcParams['figure.figsize'] = (8, 4)

plt.plot(np.arange(1,10) , np.cumsum(pca.explained_variance_ratio_))

plt.axvline(5,color='black', linestyle='--')

plt.xlabel('Number of Components')

plt.ylabel('Cummulative Explained Variance')

plt.title('Scree Plot')

plt.show();
pca = PCA(n_components=5)

pca.fit(x_scaled)

pca_loadings = pca.components_

plt.rcParams['figure.figsize'] = (8, 4)

plt.figure()

for i in range(2):

    plt.subplot(1,2,i+1)

    heights = np.squeeze(pca_loadings[i,:])

    bars = ['Rating','Reviews','Installs','Price','positive_proportion','negative_proportion','neutral_proportion','Sentiment_Polarity','Sentiment_Subjectivity']

    x_pos = np.arange(len(bars))

    plt.bar(x_pos, heights)

    plt.xticks(x_pos, bars,rotation=90)

    plt.title("Loadings PC "+str(i+1))

    plt.ylim(-1,1)

print(i)
from kmodes.kprototypes import KPrototypes

pca_scores = pca.fit_transform(x_scaled)

sentiment3[['PCA1','PCA2','PCA3','PCA4','PCA5']] = pd.DataFrame(pca_scores[:,0:5])

cluster_df = sentiment3[['PCA1','PCA2','PCA3','PCA4','PCA5','Category','Content Rating']].dropna()

kproto = KPrototypes(n_clusters=10, init='Cao')

clusters = kproto.fit_predict(cluster_df, categorical=[5, 6])

cluster_df['kmodes'] = clusters
g = sns.lmplot("PCA1", "PCA2", data=cluster_df,  

           hue='kmodes', fit_reg=False)

g.set(xlabel = 'First Principle Component',ylabel='Second Principle Component',title='Clustering of All Apps')

g._legend.set_title('Cluster')

new_labels = ['cluster 1', 'cluster 2','cluster 3','cluster 4','cluster 5','cluster 6','cluster 7','cluster 8','cluster 9','cluster 10']

for t, l in zip(g._legend.texts, new_labels): t.set_text(l);
syms = sentiment3[['App','PCA1','PCA2','PCA3','PCA4','PCA5','Category','Content Rating']].dropna().App

for s, c in zip(syms[5:15], clusters[5:15]):

    print("App: {}, cluster:{}".format(s, c))
app = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

review = pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')

app.head(3)
print ("The shape of the app dataset is " + str(app.shape))
print ("The data type of the app dataset is:\n" + str(app.dtypes))
review.head(3)
print ("The shape of the review dataset is " + str(review.shape))
print ("The data type of the review dataset is:\n" + str(review.dtypes))
print ("The unique number of observations of apps in app dataset is:" + str(app.App.nunique()))
app = app.drop_duplicates(subset=['App'], keep='first')
app = app[(app.Price != 'Everyone' )]

app.Price = app.dropna().Price.apply(lambda x: x.replace('$',''))

app['Price'] = app['Price'].astype(float)
app['Last Updated'] = pd.to_datetime(app['Last Updated'] )
app.Installs = app.dropna().Installs.apply(lambda x: x.replace('+','').replace(',',''))

app['Installs'] = app['Installs'].dropna().astype(int)
app['Reviews'] = app['Reviews'].astype(float)
app.isna().sum()
app.loc[app.Type == 'Free', 'Price'] = 0

app.Rating = app.Rating.fillna(app.Rating.mean())
review.isna().sum()
review = review.dropna(axis = 'rows') 
app.head(3)
paid = app[app['Type'].isin(['Paid'])] 

plt.rcParams['figure.figsize'] = (8, 4)

plt.hist(paid.Price,bins=500,color = "skyblue")

plt.xlim(0,40) # remove outliers

plt.xlabel('prices')

plt.ylabel('frequency')

plt.title('Distribution of prices for paid apps')

plt.show();
plt.rcParams['figure.figsize'] = (8, 4)

rat = app.Rating

plt.hist(rat,bins=60,color = "skyblue")

plt.xlabel('ratings')

plt.ylabel('frequency')

plt.title('Distribution of ratings')

plt.show();
plt.rcParams['figure.figsize'] = (8, 4)

rat = np.log(app.Installs)

plt.hist(rat,bins=60,color = "skyblue")

plt.xlabel('log(installs)')

plt.ylabel('frequency')

plt.title('Distribution of installs')

plt.show();
plt.rcParams['figure.figsize'] = (8, 4)

rew = app.Reviews.dropna()

plt.hist(np.log(rew+1),bins=60,color = "skyblue")

plt.xlabel('log(reviews)')

plt.ylabel('frequency')

plt.title('Distribution of reviews')

plt.show();
plt.rcParams['figure.figsize'] = (15, 6)

count_category = app.dropna().groupby(['Category']).count() 

count_category.App.plot(kind='bar', fontsize = 7, rot=0,title="Number of Apps in each category",color = "skyblue",alpha=0.5)

plt.ylabel('Count')

plt.xlabel('Category')

plt.xticks(rotation=90); # prevent x axis overlap
plt.rcParams['figure.figsize'] = (15, 6)

sns.boxplot(y = np.log(app['Installs']), x = app['Category'],width=0.5, linewidth=0.8,fliersize=1)

plt.xticks(rotation=90)

#plt.ylim(0,8)

plt.title('Installs in Different Categories',fontsize = 20)

plt.ylabel('Installs',fontsize = 15)

plt.xlabel('category',fontsize = 15)

plt.show();
plt.rcParams['figure.figsize'] = (8, 5)

app.groupby(['Category'])['Rating'].mean().sort_values(ascending=False).head(10).plot(kind='bar')

plt.title('Top 10 categories in rating')

plt.ylabel('average rating');
plt.rcParams['figure.figsize'] = (8, 5)

app.groupby(['Category'])['Installs'].mean().sort_values(ascending=False).head(10).plot(kind='bar')

plt.title('Top 10 categories in installs')

plt.ylabel('average installs');
plt.rcParams['figure.figsize'] = (8, 5)

app.groupby(['Category'])['Price'].mean().sort_values(ascending=False).head(10).plot(kind='bar')

plt.title('Top 10 categories in prices')

plt.ylabel('average price');
plt.rcParams['figure.figsize'] = (8, 5)

app.groupby(['App'])['Rating'].mean().sort_values(ascending=False).head(10).plot(kind='bar')

plt.title('Top 10 apps in rating')

plt.ylabel('average rating');
plt.rcParams['figure.figsize'] = (8, 5)

app.groupby(['Genres'])['Rating'].mean().sort_values(ascending=False).head(10).plot(kind='bar')

plt.title('Top 10 genres in rating')

plt.ylabel('average rating');