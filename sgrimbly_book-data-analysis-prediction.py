import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline
def load_data(file_path):

    return pd.read_csv(file_path, sep=r'\s*,\s*', header=0, engine='python', error_bad_lines=False)

books = load_data("../input/goodreadsbooks/books.csv").set_index("bookID")
books.head()
books.info()
books.describe()
import matplotlib.pyplot as plt

books.hist(bins=50, figsize=(20,15))

#weather.hist(bins=50, figsize=(20,15))

plt.show()
corr_matrix = books.corr()

corr_matrix
attributes = ["average_rating","num_pages","ratings_count","text_reviews_count"]
books[attributes].head()
# Pandas method

from pandas.plotting import scatter_matrix

scatter_matrix(books[attributes], figsize=(12,8))
# Seaborn method

sns.pairplot(data=books[attributes])

sns.set_palette("Set3")
sns.set_context('paper')

sns.set_palette("Set3")

plt.figure(figsize=(15,10))

ax = languages = books.groupby('language_code')['title'].count().plot.bar()

plt.title('Language Code')

plt.xticks(fontsize = 15)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x()-0.3, p.get_height()+100))
# Sorting by weighted factor - not ideal...weights num reviews highly!

books["weighted_rating"] = books["average_rating"]*books["ratings_count"]

books = books.sort_values(by="weighted_rating", ascending=False)
books.head()
books.plot(kind="scatter", x="num_pages", y="average_rating")#, alpha=0.1, s=AVG_speed, label="Speed", figsize=(10,7), c=trips["ETA"], cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
books.plot(kind="scatter", x="num_pages", y="weighted_rating")#, alpha=0.1, s=AVG_speed, label="Speed", figsize=(10,7), c=trips["ETA"], cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.figure(figsize=(10,10))

plot = sns.countplot(y = "authors", data = books, order = books['authors'].value_counts().iloc[:20].index, palette = "Set3")

plt.xlabel('Number of Books')

plt.ylabel('Authors')
most_rated = books.sort_values('ratings_count', ascending = False).head(20).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(most_rated['average_rating'], most_rated.index, palette='Set3')
sns.distplot(books['average_rating'], 

             bins = 80,

             kde_kws={"color": "coral", "lw": 1, "label": "KDE"}, 

             hist_kws={"histtype": "stepfilled", "linewidth": 1, "alpha": 1, "color": "skyblue"});

plt.xlabel("Average Rating")
sns.distplot(books['num_pages'], 

             bins=100,

             kde_kws={"color": "coral", "lw": 1, "label": "KDE"}, 

             hist_kws={"histtype": "stepfilled", "linewidth": 1, "alpha": 1, "color": "skyblue"});

plt.xlabel("Book Length (pages)")

plt.xlim(0,2000)
plt.figure(figsize=(15,10))

#books.dropna(0, inplace=True)

sns.set_context('paper')

ax = sns.jointplot(x="average_rating",y='text_reviews_count', kind='scatter',  data= books[['text_reviews_count', 'average_rating']])

ax.set_axis_labels("Average Rating", "Text Review Count")

plt.show()
plt.figure(figsize=(15,10))

sns.set_context('paper')

ax = sns.jointplot(x="average_rating", y="num_pages", data = books, color = 'crimson')

ax.set_axis_labels("Average Rating", "Number of Pages")
without_outliers = books[~ (books['num_pages']>1500) & (books['ratings_count']>50) & (books['text_reviews_count']>50) & (books['num_pages']>25)]

ax = sns.jointplot(x="average_rating", y="num_pages", data = without_outliers, color = 'darkcyan')

ax.set_axis_labels("Average Rating", "Number of Pages")
worst_books = without_outliers.sort_values('average_rating').set_index('title').head(40)

plt.figure(figsize=(15,10))

ax = sns.barplot(worst_books['average_rating'], worst_books.index, palette='Set3')

ax.set(xlabel="Average Rating", ylabel="Book Title")
best_books = without_outliers.sort_values('average_rating', ascending=False).set_index('title').head(40)

plt.figure(figsize=(15,10))

ax = sns.barplot(best_books['average_rating'], best_books.index, palette='Set3')

ax.set(xlabel="Average Rating", ylabel="Book Title")
plt.figure(figsize=(15,10))

duplicate = books['title'].value_counts()[:40]

#rating = books.average_rating[:20]

ax = sns.barplot(x = duplicate, y = duplicate.index, palette='Set3')

ax.set(xlabel="Occurences", ylabel="Book Title")
# Author book ratings over time

#popular_authors = ['Stephen King', 'Agatha Christie', 'Dan Brown', 'J.K. Rowling']

#books[books['authors']==popular_authors[0]].sort_values("average_rating", ascending=False)
ratings = np.arange(0, 5.5, 0.5)

#print(len(ratings))

groups = pd.cut(books["average_rating"], bins=ratings)

grouped_ratings = books.groupby(groups).sum()

grouped_ratings

#grouped_ratings['average_rating']

#sns.distplot(grouped_ratings['ratings_count'])

#grouped_ratings.plot.hist("average_rating", figsize=(10,10), colormap="Set3")

#plt.pie(grouped_ratings["average_rating"], explode = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1), labels = ratings, autopct='%1.1f%%', shadow=True, startangle=90)

#plt.axis('equal')

#plt.show()

grouped_ratings.plot.pie(y="average_rating", figsize=(10,10), colormap="hsv")
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(train_df, train_labels)
# Split data for training

books_train = training.iloc[:-10000]

oot_df = training.iloc[-10000:]

oot_df.shape,train_df.shape

train_df = train_df.drop('Timestamp',axis=1)

train_labels = train_df["ETA"]

train_df = train_df.drop('ETA',axis=1)