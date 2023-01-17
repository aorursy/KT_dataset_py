# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (be.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os



import matplotlib.pyplot as plt



dataset_dir = '../input'

dataset = 'Amazon_Unlocked_Mobile.csv'
col_names = ['product_name', 'brand_name', 'price', 'rating', 'review', 'votes']

df = pd.read_csv(os.path.join(dataset_dir, dataset), header=0, names=col_names)



df.head()
df.shape
df.count()
df.dtypes
unique_products = df.product_name.value_counts()



unique_products.count()
unique_products[:20].plot(kind='bar')
unique_brands = df.brand_name.value_counts()



unique_brands.count()
unique_brands[:20].plot(kind='bar')
price_sorted_df = df.sort_values('price', ascending=False)



price_sorted_df.head()
df.rating.value_counts().sort_index().plot(kind='bar')

plt.xlabel('Rating')

plt.ylabel('Count')
review_length = df.review.dropna().map(lambda x: len(x))



review_length.loc[review_length < 1400].hist()

plt.xlabel('Review length (Number of character)')

plt.ylabel('Count')
vote_sorted_df = df.sort_values('votes', ascending=False)



vote_sorted_df.head()
brand_ratings = df.groupby(

    'brand_name'

    ).rating.agg(

        ['count', 'min', 'max', 'mean', 'std', 'var']

    ).sort_values(

        'count', ascending=False

    )



brand_ratings.head()
product_ratings = df.groupby(

    'product_name'

    ).rating.agg(

        ['count', 'min', 'max', 'mean', 'std', 'var']

    ).sort_values(

        'count', ascending=False

    )



product_ratings.head()
price_counts = df.price.value_counts()



plt.scatter(price_counts.index, price_counts)

plt.xlabel('Price')

plt.ylabel('# Reviews')

plt.xlim(xmin=0)

plt.ylim(ymin=0)