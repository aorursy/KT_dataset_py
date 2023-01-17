import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import scipy.stats



from sklearn.preprocessing import OrdinalEncoder
df = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines=False)
skipped_lines_percent = round(4/(df.shape[0]+4)*100,4)

skipped_lines_percent
df.head()
df.shape
df.columns.to_list()
df.columns = df.columns.str.replace(' ', '')

df.columns.to_list()
print(f'Number of columns before removing: {df.shape[1]}')

df = df.drop(['bookID', 'isbn', 'isbn13', 'title', 'authors'], axis=1)

print(f'Number of columns after removing: {df.shape[1]}')
df.columns.to_list()
df.nunique()
df.language_code.unique()
df.language_code = df.language_code.replace(to_replace ='en-..', value = 'eng', regex = True)

np.sort(df.language_code.unique())
before = df.language_code.unique()



enc = OrdinalEncoder()

df.language_code = enc.fit_transform(df.language_code.values.reshape(-1, 1)).astype(int)
pd.DataFrame(data={'before': before,

                   'after': df.language_code.unique()}).sort_values(by='before')
df['year'] = df.publication_date.str.rsplit("/", n=3, expand=True)[2].astype(int)

# n=3 because value is splitted into 3 parts: day, month and year

# [2] because we are interested only in 'year'



df.head(2)
df = df.drop(['publication_date'], axis=1)

df.head(2)
df.describe()
df.agg(['mean', 'median', 'kurtosis', 'skew']).T
results = []

p_value_list = []

alpha = 0.05



for i in df._get_numeric_data().columns:

    p_value = scipy.stats.normaltest(df[i])[1] # to get only p_value without a statistic

    p_value_list.append(p_value)

    if p_value < alpha:

        results.append('rejected')

    else:

        results.append('not rejected')

        

pd.DataFrame(data={'variable': df._get_numeric_data().columns,

                    'p_value': p_value_list,

                    'null hypothesis': results})
f, axes = plt.subplots(3,2, figsize=(15, 10))

sns.distplot(df.average_rating, color='skyblue', ax=axes[0, 0])

sns.distplot(df.num_pages, color='olive', ax=axes[0, 1])

sns.distplot(df.ratings_count, color='gold', ax=axes[1, 0])

sns.distplot(df.text_reviews_count, color='teal', ax=axes[1, 1])

sns.distplot(df.year, color='skyblue', ax=axes[2, 0])

sns.countplot(x = 'language_code', data = df, ax=axes[2,1])

plt.show()