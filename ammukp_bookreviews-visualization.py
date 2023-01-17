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
ratings=pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv',sep=';',encoding='latin-1')

users=pd.read_csv("../input/bookcrossing-dataset/Book reviews/BX-Users.csv",sep=';',encoding='latin-1')

books = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Books.csv', sep=';',encoding='latin-1',names=['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l'],low_memory=False, skiprows=1)



ratings=ratings.rename(columns={'ISBN':'isbn'})

df1 = pd.merge(users, ratings, on='User-ID')

df1 = pd.merge(df1, books, on='isbn')
yeardata=df1['year_of_publication'].value_counts().reset_index()

yeardata.columns=['year','bookspublished']

yeardata['year']='year '+yeardata['year']



yeardata_sorted=yeardata.sort_values(by='bookspublished',ascending=False)

yeardata_sorted=yeardata_sorted.head(25)

yeardata_sorted.plot.bar(x='year',y='bookspublished',color='green')
yeardata_sorted=yeardata_sorted[yeardata_sorted['year']!='year 0']

yeardata_sorted.plot.bar(x='year',y='bookspublished')
dp=df1['publisher'].value_counts().reset_index()

dp.columns=['publisher','count']

dp.sort_values(by='count',ascending=False)

publisher_sorted=dp.head(10)

publisher_sorted.plot.bar(x='publisher',y='count',color='red')
authors=df1['book_author'].value_counts().reset_index()

authors.columns=['Author','NumOfBooks']

authors.sort_values(by='NumOfBooks',ascending=False).head(20).plot.bar(x='Author',y='NumOfBooks')
booksnum=df1['book_title'].value_counts().reset_index()

booksnum.columns=['BookTitle','Count']

booksnum.sort_values(by='Count',ascending=True).tail(10).plot.barh(x='BookTitle',y='Count',color='Purple')
import plotly.express as px

booksnum=df1['book_title'].value_counts().reset_index()

booksnum.columns=['BookTitle','Count']

booksnum=booksnum.sort_values(by='Count',ascending=True).tail(10)



fig1=px.bar(booksnum, y="BookTitle", x="Count",orientation='h')

fig1.update_traces(marker_color='olivedrab')

fig1.show()



#USING ONE OF THE COLUMN VALUES TO DEFINE COLOR

fig = px.bar(booksnum, y="BookTitle", x="Count",orientation='h',color='Count')

fig.show()

books_rated=df1.groupby(['book_title']).sum().reset_index()

books_rated.sort_values(by='Book-Rating',ascending=True).tail(20).plot.barh(x='book_title',y='Book-Rating',color='Pink')
df2=df1

df2=df2.rename(columns={'Book-Rating':'BookRating'})



grp3=df2.groupby('Location').BookRating.mean().to_frame()

grp3.reset_index()



#pandas.query('column_name.str.contains("abc")')

grp4=grp3.query('Location.str.contains("usa")',engine='python').sort_values(by='Location',ascending=True).head(20).reset_index()

grp4=grp4.rename(columns={'Location':'Locality','BookRating':'Ratings'})

grp4.plot.bar(x='Locality',y='Ratings',Color='Gold')

#SAME DATA ANOTHER VISUALIZATION

diag = px.bar(grp4, x="Ratings", y="Locality",orientation='h',color='Locality')

diag.show()
##last word of the location column using regular expression and stored in other column named Country 

df2['Country'] = df2.Location.str.extract(r'\b(\w+)$', expand=True)

df2.head()

t1=df2.groupby('Country').BookRating.mean().reset_index()

countries=['usa','germany','turkey','africa','zimbabwe','yugoslavia','albania']

t1=t1[t1.Country.isin(countries)]

t1['Country']=t1.Country.str.capitalize() #capitalize first letter

t1.plot.bar(x='Country',y='BookRating',color='teal')
