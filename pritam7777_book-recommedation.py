# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
u_cols= ['user_id', 'location', 'age']

users=pd.read_csv("/kaggle/input/bookcrossing-dataset/Book reviews/BX-Users.csv",sep=';',names= u_cols,encoding='latin-1',low_memory=False)





i_cols = ['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']

item=pd.read_csv('/kaggle/input/bookcrossing-dataset/Book reviews/BX-Books.csv',sep=';',names=i_cols,encoding='latin-1',low_memory=False)



r_cols = ['user_id', 'isbn', 'rating']

ratings = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1',low_memory=False)
users.head()
item.head()
ratings.head()
df=pd.merge(users,ratings,on='user_id')

df.head()
df=pd.merge(df,item,on='isbn')

df.head()
df=df.drop(columns=['img_s','img_m','img_l'])

df=df.drop( df.index[0])

df.head()
df.isnull().sum()
df=df.dropna()

df.drop_duplicates(inplace=True)
df['location'].value_counts().to_frame()
df['publisher'].value_counts().to_frame()
df['year_of_publication'].value_counts()
df.year_of_publication.unique()
wrong_values = df[df['year_of_publication'].isin(['\\"Freedom Song\\""', 'John Peterman', '2030', 'Frank Muir', 'Isadora Duncan', '2050', 'Karen T. Whittenburg', 

                                                  'ROBERT A. WILSON', '2038', 'George H. Scherr', 'Stan Berenstain', '2026', 'Francine Pascal', '2021', 'Gallimard',

                                                  'DK Publishing Inc', '2037', 'Luella Hill', 'Salvador de Madariaga', 'K.C. Constantine', 'Bart Rulon', 'Alan Rich',

                                                  'Jules Janin', '2024'])].index

df.drop(wrong_values,inplace=True)
df['user_id'] = df['user_id'].astype('int')

df['age'] = df['age'].astype('int')

df['rating'] = df['rating'].astype('int')

df['year_of_publication'] = df['year_of_publication'].astype('int')



df.dtypes
df.describe()
df.corr()
import matplotlib.pyplot as plt

import seaborn as sns
sns.pairplot(df)
df_best_authors=df.groupby(by=['book_author','book_title']).mean().sort_values (by=['rating'],ascending=False)

df_best_authors.head(10)
from sklearn.preprocessing import StandardScaler

clus_df = df[['age', 'rating', 'year_of_publication']]

X=clus_df.values[:,1:]

X=np.nan_to_num(X)

clus_df=StandardScaler().fit_transform(X)

clus_df


from sklearn.cluster import KMeans



wcss = []



for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 1)

    kmeans.fit(clus_df)

    

    wcss.append(kmeans.inertia_)


plt.figure(figsize = (12, 8))

sns.lineplot(range(1, 11), wcss, marker = 'o', color = 'darkorchid')



plt.xticks(fontsize = 14)

plt.yticks(fontsize = 14)

plt.title('The Elbow Method', fontsize = 18)

plt.xlabel('Number of clusters', fontsize = 16)

plt.ylabel('Within Cluster Sum of Squares', fontsize = 16)



plt.show()


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 1)

kmeans.fit(X)

y_kmeans = kmeans.predict(X)