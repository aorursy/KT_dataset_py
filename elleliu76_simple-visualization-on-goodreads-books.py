import warnings

warnings.filterwarnings('ignore')

import pandas as pd

from matplotlib import pyplot as plt

plt.style.use('ggplot')

df = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines = False)

df.head()
df['bookID'].nunique()
df.info()
df['publication_date'] = pd.to_datetime(df['publication_date'], format='%m/%d/%Y', errors='coerce')

df.info()
tempDf = df.loc[df['ratings_count'] >= 50]

hi_avg_rating = tempDf.groupby(['title','publisher'])['average_rating'].max().sort_values(ascending=False).head(5).reset_index()

hi_avg_rating
fig, ax = plt.subplots(figsize = (5,3))

ax.set_title('Top 5 books with their ratings', fontsize = 12, weight = 'bold')

ax.set_ylabel('Book Title')

ax.set_xlabel('Average ratings')

ax.barh(hi_avg_rating['title'],hi_avg_rating['average_rating'])

for i in ['top','bottom','left','right']:

    ax.spines[i].set_visible(False)

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')

ax.grid(False)

ax.set_facecolor("white")

ax.invert_yaxis()

for j in ax.patches:

    plt.text(j.get_width()+0.2, j.get_y()+0.5, 

             str(round((j.get_width()),2)), fontsize = 10, color = 'black')

plt.show()
df_pub = df.loc[df['publication_date'] >= '01/01/1990']

df_pub['publication_year'] = pd.DatetimeIndex(df_pub['publication_date']).year

df_pub.shape
df_pub_temp = df_pub.groupby(['publication_year'])['bookID'].count().reset_index()

df_pub_temp
df_pub_temp.describe()
fig_pub, ax_pub = plt.subplots()

fig_pub = plt.figure(figsize = (5,3))

ax_pub.set_title('Number of published books since 1990 ', fontsize = 12, weight = 'bold')

ax_pub.set_xlabel('Publication Year')

ax_pub.set_ylabel('Number of Books')

ax_pub.grid(False)

ax_pub.set_facecolor("white")

ax_pub.bar(df_pub_temp['publication_year'],df_pub_temp['bookID'])

plt.show()