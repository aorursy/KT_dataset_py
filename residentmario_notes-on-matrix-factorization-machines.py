# import pandas as pd
# pd.set_option('max_columns', None)
# ratings = pd.read_csv("../input/goodbooks-10k/ratings.csv")
# books = pd.read_csv("../input/goodbooks-10k/books.csv")
# len(ratings), len(books)
# import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')

# ratings['user_id'].value_counts().plot.hist(bins=50, title='N(Recommendations) per User', figsize=(12, 6))
# (books
#      .drop(['id', 'best_book_id', 'work_id', 'authors', 'original_title', 'title', 'language_code', 'image_url', 'small_image_url', 'isbn', 'isbn13'], axis='columns')
#      .drop(['work_ratings_count', 'work_text_reviews_count'], axis='columns')
#      .set_index('book_id')
#      .head()
# )