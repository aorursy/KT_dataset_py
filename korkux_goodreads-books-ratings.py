import pandas as pd

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False)
df.head()
df.shape
df.isna().sum(axis=0)
df.dtypes
df.where(df['title'] == 'The Known World').count()
books_with_reviews = df[df['text_reviews_count'] > 0]

books_with_no_reviews = df[df['text_reviews_count'] == 0]
books_with_no_reviews.shape[0]
ax = sns.jointplot(data= books_with_no_reviews, x='text_reviews_count', y='average_rating')

ax.set_axis_labels('Reviews Count', 'Averange Rating')

plt.title('Books with no reviews')
best_rating = books_with_reviews.groupby('title')['average_rating'].mean().sort_values(ascending=False)

best_rating = pd.DataFrame(best_rating)

best_rating
best_rating = best_rating.loc[best_rating['average_rating'] == 5.0]

best_rating = best_rating.reset_index()

print(f'total: {best_rating.shape[0]}')

best_rating
fig = px.bar(best_rating, title='Highest rated books', x='title', y='average_rating', text='average_rating', labels={'title':'Books Title', 'average_rating': 'Rating'})

fig.show()
best_reviews = books_with_reviews.groupby('title').agg({'average_rating': ['mean'], 'text_reviews_count': ['sum']})

best_reviews.columns = ['average_rating', 'text_reviews_count']

best_reviews = best_reviews.reset_index()

best_reviews = best_reviews.sort_values(by=['text_reviews_count'], ascending=False).head(10)

best_reviews
fig = px.scatter(best_reviews, x="text_reviews_count", y="average_rating", hover_data=['title'], labels={'text_reviews_count': 'Reviews Count', 'average_rating': 'Average Rating'})

fig.update_layout(title='Reviews Count vs Rating')

fig.show()