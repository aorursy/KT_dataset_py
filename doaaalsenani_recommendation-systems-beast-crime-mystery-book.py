import pandas as pd

import numpy as np

import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.display import display

pd.options.display.max_columns = None

pd.options.display.max_rows = None
book = pd.read_csv('../input/goodreads-beast-crime-mystery-books/best_crime_and_mystery_books.csv')
book.head()
book.tail()
book.shape
book['id'].value_counts().sum()
book['title'].value_counts().sum()
book['book_author'].nunique()
book.info()
missing= book.isnull().sum().sort_values(ascending=False)

percentage = (book.isnull().sum()/ book.isnull().count()).sort_values(ascending=False)

missing_recommend_book = pd.concat([missing, percentage], axis=1, keys=['Missing', '%'])

missing_recommend_book.head(8)
# visualize the location of missing values

sns.heatmap(book.isnull(),yticklabels=False, cmap='viridis', cbar=False)
duplicateRowsbook = book[book.duplicated()]
duplicateRowsbook
#Check Missing 

book.loc[book.publication_year.isnull(),:].head()
#Filling Missing 

book.publisher.fillna(value='other', inplace=True)
# replace

book['publisher'] =book['publisher'].str.replace('.','')

book['publisher'] =book['publisher'].str.replace('(','')

book['publisher'] =book['publisher'].str.replace(')','')

book['publisher'] =book['publisher'].str.replace('&','and')

book['publisher'] =book['publisher'].str.replace('-','')

book['publisher'] =book['publisher'].str.replace(',','')

book['publisher'] =book['publisher'].str.replace('/','')

book['publisher'] =book['publisher'].str.replace(';','')

book['publisher'] =book['publisher'].str.replace(':','')



book['publisher'] =book['publisher'].str.replace('المؤسسة العربية الحديثة','Modern Arab Foundation')

book['publisher'] =book['publisher'].str.replace('مؤسسة سندباد للنشر والأعلام','Sinbad Foundation for Publishing and Media')

book['publisher'] =book['publisher'].str.replace('Bricbooks http://claudebrick.wix.com/bricbooks','other')

book['publisher'] =book['publisher'].str.replace('http://www.amazon.com/While-the-Village-Sleeps-ebook/dp/B00DY9FRLG  ','other')
# replace

book['title'] =book['title'].str.replace('.','')

book['title'] =book['title'].str.replace('(','')

book['title'] =book['title'].str.replace(')','')

book['title'] =book['title'].str.replace('&','and')

book['title'] =book['title'].str.replace('-','')

book['title'] =book['title'].str.replace(',','')

book['title'] =book['title'].str.replace('/','')

book['title'] =book['title'].str.replace(';','')

book['title'] =book['title'].str.replace(':','')

book['title'] =book['title'].str.replace('#','no ')
book.loc[(book.language_code == 'ara')]
#Translate the ara title

book['title'] =book['title'].str.replace('اغتيال يوسف','One Spirit Scenario')

book['title'] =book['title'].str.replace('مخلب الشيطان',"Devil's Claw")

book['title'] =book['title'].str.replace('مذكرات اجرامية 1 قاتل المتحرشين','Criminal notes 1 Assassiners molesters')

book['title'] =book['title'].str.replace('مذكرات اجرامية 4  البطل المحتقر','Criminal memo 4 despised hero')

book['title'] =book['title'].str.replace('مذكرات اجرامية 2   البومة السوداء','Criminal diary 2 black owl')

book['title'] =book['title'].str.replace('العروس قاتلة في العصر العثماني','The bride was fatal in the Ottoman era')

book['title'] =book['title'].str.replace('مذكرات اجرامية 3  عاشق النار','Criminal notes 3 lover of fire')

book['title'] =book['title'].str.replace('سيناريو روح واحدة','Assassination of Joseph')
#Translate

book['book_author'] =book['book_author'].str.replace('محمد','Mohamed')

book['book_author'] =book['book_author'].str.replace('عادل','Adel')



book['book_author'] =book['book_author'].str.replace('نبيل','Nabil')

book['book_author'] =book['book_author'].str.replace('فاروق','Farouk')



book['book_author'] =book['book_author'].str.replace('رضا','Reda')

book['book_author'] =book['book_author'].str.replace('داليا','Dalia')

# replace

book['book_author'] =book['book_author'].str.replace('-','')

book['book_author'] =book['book_author'].str.replace('.','')
#Filling Missing by mean

book['publication_year'].fillna(book['publication_year'].mean(), inplace=True)
book.loc[(book.publication_year==6)]
book.loc[(book.publication_year==17)]
book.at[1499, 'publication_year']=2020

book.at[4210, 'publication_year']=2017

# or 

# book.set_value(1499,'publication_year','2020')

# book.set_value(4210,'publication_year','2017')



# convert type to int

book['publication_year']=book['publication_year'].astype('int')

book['publication_year'].dtype
#Filling Missing recommend_book by 0 

book.num_pages.fillna(value=0, inplace=True)
# convert type to int

book['num_pages']=book['num_pages'].astype('int')

book['num_pages'].dtype
book['publisher'] =book['publisher'].apply(lambda x:x.lower())

book['title'] =book['title'].apply(lambda x:x.lower())

book['book_author'] =book['book_author'].apply(lambda x:x.lower())

book['language_code']=book['language_code'].apply(lambda x:x.lower())
fig=plt.figure(figsize=(15,10))

ax = fig.gca()

sns.heatmap(book.corr(), annot=True,ax=ax, cmap=plt.cm.YlGnBu)

ax.set_title('The correlations between all numeric features''\n')

palette =sns.diverging_palette(80, 110, n=146)

plt.show()
ax =book.groupby('language_code')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('language_code').plot.bar(figsize=(14,8),rot=0)

plt.title('language code',fontsize=20)

plt.xticks(fontsize=15)

for p in ax.patches:

    ax.annotate(str(p.get_height()),(p.get_x()+0.1,p.get_height()+100))
most_rated = book.sort_values('ratings_count', ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(most_rated['ratings_count'], most_rated.index, palette='Spectral')

plt.title('Top 10 most rated books')

plt.show()
highly_rated_author =book[book['average_rating']>4.4]

highly_rated_author = highly_rated_author.groupby('book_author')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('book_author')

plt.subplots(figsize=(12,10))

ax = highly_rated_author['title'].sort_values().plot.barh(width=0.9,color=sns.color_palette('Spectral',12))

ax.set_xlabel("Total books ")

ax.set_ylabel("Authors")

ax.set_title("Top 10 highly rated authors")

plt.show()
ax =book.groupby('publication_year')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('publication_year').plot.bar(figsize=(14,8),rot=0)

plt.title('The top 10 publication years')

plt.xlabel('publication year')





plt.show()
sns.kdeplot(book['average_rating'], shade = True)

plt.title('Rating Distribution')

plt.xlabel('Rating')

plt.ylabel('Frequency')

plt.show()
#Printing the book title and book_author randomly

print (book['title'] [0])

print (book['book_author'][0])
book.loc[(book.book_author == 'stieg larsson')]
#Printing the book title and book_author randomly

print (book['title'] [3])

print (book['book_author'][3])
book.loc[(book.book_author == 'daphne du maurier')]
# Printing the book title and book_author randomly

print (book['title'] [7])

print (book['book_author'][7]) 
book.loc[(book.book_author == 'umberto eco')]
#Printing the book title and book_author randomly

print (book['title'] [5])

print (book['book_author'][5]) 
book.loc[(book.book_author == 'mario puzo')]
#Printing the book title and book_author randomly

print (book['title'] [9])

print (book['book_author'][9]) 
book.loc[(book.book_author == 'dennis lehane')]
def recommender_systems(title, book_author):

    

    # Matching the genre with the recommend_bookset and reset the index

    recommend_book = book.loc[book['book_author'] == book_author]  

    recommend_book.reset_index(level = 0, inplace = True) 

  

    # Convert the index into series

    index_into_series = pd.Series(recommend_book.index, index = recommend_book['title'])

    

    #Converting the book title into vectors and used bigram

    tfidfv = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df = 1)

    tfidfv_matrix = tfidfv.fit_transform(recommend_book['title'])

    



    # Calculating the similarity measures based on Cosine Similarity

    similarity_measures = cosine_similarity(tfidfv_matrix, tfidfv_matrix)

    

    # Get the index corresponding to original_title

    indxe = index_into_series[title]

    

    # Get the pairwsie similarity scores 

    similarity = list(enumerate(similarity_measures[indxe]))

    

    # Sort the books

    similarity = sorted(similarity, key=lambda x: x[1], reverse=True)

    

    # Scores of the most similar books    

    similarity = similarity[1:]

    

    # Book indicies

    book_index_into_series = [i[0] for i in similarity]

   

    # book recommendation

    recommendation= recommend_book[['title']].iloc[book_index_into_series]

    return recommendation
print(recommender_systems('the girl with the dragon tattoo millennium no 1','stieg larsson'))
print(recommender_systems('rebecca','daphne du maurier'))
print(recommender_systems('the name of the rose','umberto eco'))
print(recommender_systems('the godfather','mario puzo'))
print(recommender_systems('shutter island','dennis lehane'))