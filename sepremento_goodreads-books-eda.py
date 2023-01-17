import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import warnings



from matplotlib.ticker import FormatStrFormatter



from scipy.stats import zscore

from scipy.stats import norm

from scipy.stats import f_oneway

from statsmodels.stats.multicomp import pairwise_tukeyhsd



from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import silhouette_score



from IPython.display import display



plt.style.use('seaborn-talk')  # nice readable plot style

warnings.filterwarnings('ignore')
brows = [3349,4703,5878,8980]  # bad or corrupt row, extra comma in the 'Authors' field

raw = pd.read_csv('../input/goodreadsbooks/books.csv', skiprows=brows)

raw.rename(columns={'  num_pages':'num_pages'}, inplace=True)  # there is a problem with column name, extra spaces.



random_indexes = np.random.choice(raw.shape[0], size=2000, replace=False)  # create random sample of books to explore

toy = raw.loc[random_indexes,:]  # create toy dataframe to wrangle
## DATA PREPARATION ##



data = raw.copy()  # when I finish the analysis just change from 'toy' to 'raw'



data.reset_index(inplace=True, drop=True)

# create new feature with number of authors

data['n_authors'] = [len(s) for s in data['authors'].str.split('/')]

# normalize several features

data['log_ratings'] = np.log1p(data['ratings_count'])

data['log_reviews'] = np.log1p(data['text_reviews_count'])

data['log_pages'] = np.log1p(data['num_pages'])



# create categorical feature:

data['bins_pages'] = pd.cut(data['log_pages'], bins=5, labels=['tiny','small','average','large','huge'])



# transform publication_date from string to three integer columns

dates = data['publication_date'].str.split('/', expand=True).astype(int)

dates.columns = ['pub_month','pub_day', 'pub_year']

data = pd.concat((data,dates),axis=1)

# drop the old 'publication_date' feature

data.drop(['publication_date'],axis=1, inplace=True)



display(data.head())
print("There are {} books in the dataframe.".format(data.shape[0]))

print("Maximum number of authors for one book is {0} for {1}.".format(data['n_authors'].max(), 

                                                                     data.loc[data['n_authors'].idxmax(), 'title']))

print("Maximum number of pages is in {1} with {0} pages.".format(data['num_pages'].max(),

                                                                 data.loc[data['n_authors'].idxmax(), 'title']))
# a dive into small books

small_books = data.loc[data['num_pages'] <=50,:]

print("There are a total of:",small_books.shape[0], "small books")



# some publisher names contain the word 'Audio'

audio = small_books.loc[small_books['publisher'].str.contains('Audio'),:].shape[0]

print("At least", audio, "of them are audiobooks, i.e. their publishers have a word 'Audio' in their name")
# books distribution by publication year

sns.countplot(data['pub_year'], palette='inferno')

plt.xlabel('Publication Year')

plt.ylabel('Published Books')

plt.title('Published Books by Year')

fig = plt.gcf()

fig.set_size_inches(15,8)

plt.xticks(rotation=75, size=8)

plt.show()
data_17 = data.loc[(data['pub_year'] >= 1989) & (data['pub_year']<=2006),:]  # subsample of data for 17 years

print('After all we are left with {} samples from years 1989 to 2006.'.format(data_17.shape[0]))
# crosstable with monthly fractions of yearly published books

pd.crosstab(data_17['pub_year'], data_17['pub_month'], 

            normalize='index').plot(kind='bar',stacked=True,cmap='tab20c', width=1)

plt.legend(bbox_to_anchor=(-0.01, -0.2), loc='upper left', ncol=6, 

           labels=['January','February','March','April','May','June','July',

                   'August','September','October','November','December'])

plt.xlabel('Publication Year')

plt.ylabel('Fraction of monthly published books')

plt.title('Published books by month and year')

f = plt.gcf()

f.set_size_inches(15,7)
# books distribution by publication month

sns.countplot(data_17['pub_month']);

plt.xlabel('Publication month')

plt.ylabel('Count')

plt.xticks(np.arange(12), labels=['January','February','March','April','May','June','July',

                                  'August','September','October','November','December'], rotation=75)

plt.title('Books published each month')

f = plt.gcf(); f.set_size_inches(15,7)



plt.show()
by_month = pd.crosstab(index=data_17['pub_year'], columns=data_17['pub_month'], normalize='index')

pval = f_oneway(by_month[1],by_month[2],by_month[3],

                by_month[4],by_month[5],by_month[6],

                by_month[7],by_month[8],by_month[9],

                by_month[10],by_month[11],by_month[12])

print('Null hypothesis is: There is no difference between aforementioned sample means')

print("Statistic:", np.round(pval[0],4))

print("p-value:", pval[1])

print('Null hypothesis REJECTED' if pval[1]<=0.05 else "Null hypothesis ACCEPTED")
cm = by_month.melt()  # making one column dataframe

tukey_results = pairwise_tukeyhsd(cm['value'], cm['pub_month'], 0.05)

tukey_results.plot_simultaneous(comparison_name=8,

                                xlabel='Mean Fraction of Books Per Month',

                                ylabel='Month')

f = plt.gcf(); f.set_size_inches(15,7)

plt.show()
# book average rating by year and average book size by year

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,5))



data_17.groupby(by='pub_year')['average_rating'].mean().plot(kind='barh', width=1, edgecolor='white',ax=ax1);

data_17.groupby(by='pub_year')['num_pages'].mean().plot(kind='barh',width=1, edgecolor='white',ax=ax2);



ax1.set_ylabel('Publication Year')

ax2.set_ylabel('')

ax2.set_yticklabels('')

ax1.set_xlabel('Average Rating')

ax2.set_xlabel('Book Size')



plt.suptitle('Other Yearly Distributions', y=1.04, size=18, weight='bold')

f = plt.gcf(); f.set_size_inches(15,5)

plt.tight_layout()

plt.show()
# distribution of books by rating

plt.hist(data['average_rating'], bins=40, edgecolor='white', color='lightcoral')

plt.xlabel('Book Average Rating')

plt.ylabel('Count')

plt.title('Book Distribution by Rating')

f = plt.gcf(); f.set_size_inches(15,7)

plt.show()
f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,14))



# distribution of books by size

ax1.hist(data_17['num_pages'], bins=40, edgecolor='darkgreen', color='palegreen')

ax1.set_xlabel('Number of Pages')

ax1.set_ylabel('Count')



# distribution of books by size on a logarithmic scale

ax2.hist(data_17['log_pages'], bins=40, edgecolor='white', color='forestgreen')

maximum = data_17['log_pages'].value_counts(bins=40).idxmax().mid

ax2.axvline(ymin=0,ymax=1, x=maximum, ls='--', c='indigo',lw=2,

            label=maximum)

ax2.legend()

ax2.set_xlabel('Number of Pages Logarithm')

ax2.set_ylabel('Count')



plt.suptitle('Books Distribution by Number of Pages', y=1.03,

             fontweight='bold', fontsize=18)

plt.tight_layout()

plt.show()
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,7))



# we have some outliers both on ratings and text reviews counts

ax1.scatter(data_17['ratings_count'], data_17['text_reviews_count']);

ax1.set_title("with outliers")

ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

ax1.tick_params('x', labelrotation=45)

ax1.set_xlabel('Ratings Count')

ax1.set_ylabel('Text Reviews Count')





# plot with removed outliers

ax2.scatter(data_17.loc[zscore(data_17['ratings_count'])<3, 'ratings_count'],

            data_17.loc[zscore(data_17['ratings_count'])<3, 'text_reviews_count']);

ax2.set_title("without outliers")

ax2.tick_params('x', labelrotation=45)

ax2.set_xlabel('Ratings Count')

ax2.set_ylabel('Text Reviews Count')



plt.suptitle('Ratings Count vs Text Reviews Count', y=1.03,

             fontweight='bold', fontsize=18)

plt.tight_layout()

plt.show()
plot_data = data_17.loc[zscore(data_17['ratings_count'])<3,['ratings_count','text_reviews_count','average_rating']]

plot_data['rating_quartile'] = pd.qcut(plot_data['average_rating'], q=4, labels=['1','2','3','4'])



g = sns.relplot(kind='scatter', data=plot_data, alpha=0.5,

                x='ratings_count',y='text_reviews_count',

                hue='rating_quartile',)

g._legend.texts[0].set_text("")

g._legend.set_title("Rating Quartile")

g._legend.set_bbox_to_anchor([0.23,0.73])



g.ax.set_xlabel('Ratings Count')

g.ax.set_ylabel('Text Reviews Count')

g.ax.set_title('Book Ratings and Text Reviews')

f = plt.gcf(); f.set_size_inches(15,7)

plt.show()
# plot only outliers

plt.scatter(data.loc[zscore(data['ratings_count'])>=3, 'ratings_count'],

            data.loc[zscore(data['ratings_count'])>=3, 'text_reviews_count']);

plt.title('Outliers Plot')

plt.xlabel('Ratings Count')

plt.ylabel('Text Reviews Count')

ax = plt.gca()

ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

data[zscore(data['ratings_count'])>=3].sort_values(by='ratings_count', ascending=False).head(10)
labels=['Average Rating','Ratings Count','Ratings Count Logarithm', 

        'Text Reviews Count','Number of Pages','Number of Authors']



f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,15))

# correlation heatmap

corr_df = data_17[['average_rating','ratings_count','log_ratings','text_reviews_count','num_pages','n_authors']].corr()

sns.heatmap(corr_df, annot=True, vmax=0.5, fmt='.2f', cmap='viridis', ax=ax1, 

            linewidth=0.5,

            xticklabels='', yticklabels=labels);

ax1.set_title('All books')



# what about books with reasonable amount of ratings?

corr_df = data_17.loc[data_17['log_ratings']> 4.6, ['average_rating','ratings_count','log_ratings','text_reviews_count','num_pages','n_authors']].corr()

sns.heatmap(corr_df, annot=True, vmax=0.5, fmt='.2f', cmap='RdYlGn', ax=ax2, 

            linewidth=0.5,

            xticklabels=labels, yticklabels=labels);

ax2.set_title('Books with more than 100 ratings')



plt.suptitle('Correlations In Data', y=1.04, weight='bold',size=18)

plt.tight_layout()

plt.show()
# connections between average rating, ratings count and book size category

colors = data_17.loc[data_17['log_ratings']>4.6, 'bins_pages'].map({'tiny':'violet',

                                                              'small':'cyan',

                                                              'average':'red',

                                                              'large':'forestgreen',

                                                              'huge':'blue'})

plot_data = data_17.loc[data_17['log_ratings']>4.6]

f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,11))

sns.violinplot(y='bins_pages', x='average_rating', data=plot_data, ax=ax1)

ax1.set_xlabel('Average Rating')

ax1.set_ylabel('Book Size Category')



sns.violinplot(y='bins_pages', x='log_ratings', data=plot_data, ax=ax2)

ax2.set_xlabel('Ratings Count Logarithm')

ax2.set_ylabel('Book Size Category')



plt.suptitle('Book Size and Ratings when Ratings Count > 100', weight='bold',size=18, y=1.04)

plt.tight_layout()

plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,7))



ax1.hist(data['ratings_count'], bins=20, )

ax1.set_title('Ratings Histogram')

ax1.set_xlabel('Number of ratings')

ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

ax1.set_ylabel('Book count')



ax2.hist(data['log_ratings'], bins=20, edgecolor='white')

ax2.set_title('Ratings\' Logarithms Histogram')

ax2.set_xlabel('Logarithm of Ratings Count')



plt.suptitle('Book Ratings Count Distributions', y=1.04, weight='bold', size=18)

plt.tight_layout()

plt.show()
plt.scatter(data_17['log_ratings'], data_17['average_rating'], alpha=0.5);

plt.xlabel('Ratings Count Logarithm')

plt.ylabel('Average Rating')

f = plt.gcf()

f.set_size_inches(15, 4)
# zero rating count

data.loc[data['log_ratings'] == 0].head()
print("There are {} unrated books in our dataframe".format(data.loc[data['log_ratings'] == 0].shape[0]))
# list of "bad" books

bbooks = data.loc[(data['log_ratings'] > 4.6) & (data['average_rating'] < 3),:]

bbooks.head()
print('There are {} "bad" books in our dataframe.'.format(bbooks.shape[0]))

print('Their average rating of "bad" books is {0:5.2f}'.format(bbooks['average_rating'].mean()))
authors = data['authors'].str.split('/')  # create Series object with lists of authors per each book

authors = authors.values  # transform to flattened numpy array of lists

authors = np.concatenate(authors)  # transform to one list

authors = np.unique(authors)  # get uniques

print("There are {} authors in our base".format(len(authors)))
# how many books did the author (co)write

num_books = [data_17[data_17['authors'].str.contains(author)]['title'].count() for author in authors]

# how many publishers did the author work with

num_publishers = [data_17[data_17['authors'].str.contains(author)]['publisher'].count() for author in authors] 

# what books did the author write

book_indexes = [data_17[data_17['authors'].str.contains(author)]['bookID'].ravel().tolist() for author in authors]

# what publishers did he or she work with

pub_names = [data_17[data_17['authors'].str.contains(author)]['publisher'].ravel().tolist() for author in authors]



total_pages = [data_17[data_17['authors'].str.contains(author)]['num_pages'].sum() for author in authors]

mean_pages = [data_17[data_17['authors'].str.contains(author)]['num_pages'].mean() for author in authors]

avg_rating = [data_17[data_17['authors'].str.contains(author)]['average_rating'].mean() for author in authors]



# in what year was the first book published

first_book = [data_17[data_17['authors'].str.contains(author)]['pub_year'].min() for author in authors]

# in what year was the latest book published

latest_book = [data_17[data_17['authors'].str.contains(author)]['pub_year'].max() for author in authors]
authors_df = pd.DataFrame({'author_name': authors, 

                           'num_books': num_books,

                           'num_publishers': num_publishers,

                           'tot_pages': total_pages,

                           'avg_pages': mean_pages,

                           'avg_rating': avg_rating,

                           'first': first_book,

                           'latest': latest_book,

                           'publishers': pub_names,

                           'book_ids': book_indexes})

display(authors_df.head())
print('{0} wrote most number of books, {1}'.format(authors_df.loc[authors_df.num_books.idxmax,'author_name'],authors_df.num_books.max()))

print('{0} (co)authored books ending up with most total pages, {1}'.format(authors_df.loc[authors_df.tot_pages.idxmax,'author_name'],authors_df.tot_pages.max()))

print('{0} worked with most number of publishers, {1}'.format(authors_df.loc[authors_df.num_publishers.idxmax,'author_name'],authors_df.num_publishers.max()))
plot_data = authors_df.sort_values(by='tot_pages', ascending=False)[['author_name', 'num_books','tot_pages']][:20]



plt.bar(x=plot_data['author_name'], height=plot_data['tot_pages'])

plt.title('Most Prolific Authors')

plt.ylabel('Total Number of Published Pages')

plt.xticks(rotation=90)

f = plt.gcf()

f.set_size_inches(15,6)

plt.show()
labels = ['Number of Books','Number of Publishers','Total Pages','Average Pages per Book','Average Rating per Book']

corr_df = authors_df[['num_books','num_publishers','tot_pages','avg_pages','avg_rating']].corr()

sns.heatmap(corr_df, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, 

            linewidth=0.5, cmap='viridis');

ax = plt.gca()

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

         rotation_mode="anchor")

plt.show()
# top 20 publishers distribution by book count

data_17['publisher'].value_counts()[:20].plot(kind='bar', width=0.8)

f = plt.gcf()

f.set_size_inches(15,7)

plt.title("Top 20 publishers")

plt.ylabel('Published Books')

plt.xlabel('Publisher')

plt.show()
print("There are {} publishers in our dataframe".format(data['publisher'].value_counts().shape[0]))

print("Top 20 publishers published {} books".format(data['publisher'].value_counts()[:20].sum()))

print("So, top 20 publishers are {0:1.2f}% of all publishers".format(20/data['publisher'].value_counts().shape[0]*100))

a = data['publisher'].value_counts()[:20].sum() / data.shape[0] * 100

print("However, they published {0:2.2f}% of all books in our dataset".format(a))
a = data['publisher'].value_counts()[:20].sum() / data.shape[0]

p = 20/data['publisher'].value_counts().shape[0]

c = ['royalblue','mistyrose']

plot_data = pd.DataFrame({'publisher':['Top 20', 'All Other'],

                          'proportion': [p,1-p],

                          'share':[a,1-a]})



f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,5))



ax1.pie(plot_data['proportion'], labels=plot_data['publisher'], colors=c,explode=[0.1,0],

        wedgeprops={'linewidth': 1, 'edgecolor':'k'})

ax1.set_title('Publishers')

ax2.pie(plot_data['share'], labels=plot_data['publisher'], colors=c, explode=[0.1,0],

        wedgeprops={'linewidth': 1, 'edgecolor':'k'})

ax2.set_title('Books Published')



plt.suptitle('Distribution of publishers',y=1.04, weight='bold', size=18)

plt.tight_layout()

plt.show()
n = int(np.round(data['publisher'].value_counts().shape[0]*0.2))

a = data['publisher'].value_counts()[:n].sum() / data.shape[0]*100

print("Top 20% of publishers published {:5.2f}% of books.".format(a))

a = data['publisher'].value_counts()[:n].sum() / data.shape[0]

p = int(np.round(data['publisher'].value_counts().shape[0]*0.2)) / data['publisher'].value_counts().shape[0]

c = ['royalblue','honeydew']

plot_data = pd.DataFrame({'publisher':['Top 20%', 'All Other'],

                          'proportion': [p,1-p],

                          'share':[a,1-a]})



f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,5))



ax1.pie(plot_data['proportion'], labels=plot_data['publisher'], colors=c,explode=[0.1,0],

        wedgeprops={'linewidth': 1, 'edgecolor':'k'})

ax1.set_title('Publishers')

ax2.pie(plot_data['share'], labels=plot_data['publisher'], colors=c, explode=[0.1,0],

        wedgeprops={'linewidth': 1, 'edgecolor':'k'})

ax2.set_title('Books Published')



plt.suptitle('Distribution of publishers',y=1.04, weight='bold', size=18)

plt.tight_layout()

plt.show()
to_cluster = data_17[['average_rating', 'num_pages', 'ratings_count', 'n_authors']]  # take the data for the 17 years

to_cluster = to_cluster.loc[(zscore(to_cluster['ratings_count'])<3) &  # remove the outliers in ratings count

                            (zscore(to_cluster['n_authors'])<3)     &  # --//-- in number of authors

                            (zscore(to_cluster['num_pages'])<3)].reset_index(drop=True)  # --//-- in number of pages

scaler = MinMaxScaler()  # scale the data

preprocessed = scaler.fit_transform(to_cluster)
# a metric for elbow method to determine best number of clusters

Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(preprocessed)

    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, marker='x')

plt.xlabel('k')

plt.ylabel('Sum of squared distances')

plt.title('Elbow Method For Optimal k', weight='bold')

plt.grid()

plt.show()
km = KMeans(n_clusters=7)

km = km.fit(preprocessed)

clusters = km.predict(preprocessed)

silh = silhouette_score(preprocessed, clusters)

print("Average cluster silhouette score is: {0:5.3f}".format(silh))
clusters = pd.Series(clusters, name='cluster', dtype='category')

plot_data = pd.concat((to_cluster, clusters), axis=1)



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

sns.scatterplot(x='average_rating',y='num_pages',hue='cluster',data=plot_data, ax=ax1)

sns.scatterplot(x='ratings_count',y='num_pages',hue='cluster',data=plot_data, ax=ax2)

sns.scatterplot(x='average_rating',y='n_authors',hue='cluster',data=plot_data, ax=ax3, alpha=0.5)

sns.scatterplot(x='ratings_count',y='n_authors',hue='cluster',data=plot_data, ax=ax4, alpha=0.5)



ax1.set_xlabel('')

ax2.set_xlabel('')

ax2.tick_params('x', labelrotation=60)

ax3.set_xlabel('Average Rating')

ax4.set_xlabel('Ratings Count')

ax4.tick_params('x', labelrotation=60)



ax1.set_ylabel('Number of Pages')

ax2.set_ylabel('')

ax3.set_ylabel('Number of Authors')

ax4.set_ylabel('')

plt.suptitle('Clusters of Books', y=1.04, weight='bold', size=18)

plt.tight_layout()

plt.show()