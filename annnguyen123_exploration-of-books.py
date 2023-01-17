import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from newspaper import Article
df = pd.read_csv("../input/books.csv", error_bad_lines=False)
df.index = df['bookID']
df.head()
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
# Correct the name of author of Harry Potter to J.K. Rowling

df.loc[df['title'].str.contains('Harry Potter', regex=True), 'correct_authors'] = 'J.K. Rowling' 

df.loc[df['correct_authors'].isna(), 'correct_authors'] = df['authors']
# Statistics about books:

statistics = df.agg({'average_rating': 'mean', '# num_pages': 'mean', 'ratings_count': 'mean', 'text_reviews_count': 'mean'})

statistics
# Which are the authors with the most book?

sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(7, 5))



# Load dataset

top_10_authors = df.groupby('correct_authors').agg({'average_rating': 'mean', '# num_pages': 'mean', 'ratings_count': 'mean', 'text_reviews_count': 'mean',

                                    'bookID' : 'nunique'}).reset_index().sort_values("bookID", ascending=False).head(10)



# Plot top 10 authors with most books

sns.set_color_codes("pastel")

sns.barplot(x="bookID", y="correct_authors", data=top_10_authors,

            label="Total", color="b").set_title('Top 10 authors with most books', weight='bold')

ax.set(xlabel='Number of books', ylabel='Authors')

# Which are the books with highest rating of Top 10 authors?

top_10_authors_info = df[df['correct_authors'].isin(top_10_authors['correct_authors'].tolist())]

top_10_authors_info['max_rating'] = top_10_authors_info['correct_authors'].map(df.groupby(['correct_authors'])['average_rating'].max())

highest_rating_top_10a = (

    top_10_authors_info[['title','correct_authors', 'language_code', 'average_rating']]

    [top_10_authors_info['average_rating']==top_10_authors_info['max_rating']]

    .sort_values("correct_authors", ascending=False))

# add heatmaps for column average_rating

cm = sns.light_palette("green", as_cmap=True)

highest_rating_top_10a = highest_rating_top_10a.style.background_gradient(cmap=cm)

highest_rating_top_10a
# What is the distribution of books across languages?

sns.set_context('notebook')

plt.figure(figsize=(15,7))

ax = df.groupby("language_code")['bookID'].nunique().plot.bar()

plt.title('Language Code', weight = 'bold')

plt.xticks(fontsize = 15)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x()-0.3, p.get_height()+100))

# What is the distribution of books across rating?

sns.set_context('notebook')

plt.figure(figsize=(15,7))

by_rating = pd.DataFrame({'num of books' : df.groupby('average_rating')['bookID'].nunique()}).reset_index()

ax = sns.lineplot(x="average_rating", y='num of books', data=by_rating)

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
# Which are the top 10 most rated books?

sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(7, 5))



# Load dataset

top_10_rated = df.sort_values("ratings_count", ascending=False).head(10)

top_10_rated

# Plot top 10 authors with most books

sns.barplot(x="ratings_count", y="title", data=top_10_rated

            , palette='rocket').set_title('The top 10 most rated books', weight='bold')

ax.get_xaxis().set_major_formatter(

    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# Alternate scraping solution, when both the API(s) fails

def html(isbn):

    url = 'https://isbndb.com/book/'+isbn

    article = Article(url)

    #article = 'https://isbndb.com/book/9780450524684'

    article.download()

    article.parse()

    ar = article.html

    ar = ar[9300:9900]

    return ar



y = html("0439785960")