#importing libraries

import pandas as pd 

import seaborn as sn

import matplotlib.pyplot as plt

import numpy as np

import numpy as np

from sklearn.linear_model import LinearRegression

import scipy.cluster.hierarchy as shc

%matplotlib inline
#Reading dataset with pandas

books= pd.read_csv('../input/goodreads-bookscsv/goodreads_books.csv')

books.head()
books.shape
books.info()
books.nunique()
#make copy of the data to start cleaning

df= books.copy()
#changing the type of The average rating colum

df['average_rating']= pd.to_numeric(df.average_rating, errors='coerce')

df['  num_pages']= pd.to_numeric(df['  num_pages'], errors='coerce')

#testing 

df.info()
#find out the zeros rows

df.isnull().sum()
#dropping unnecessary column

df.drop(['isbn','isbn13','Unnamed: 12'],axis =1,inplace=True)

#drop rows with zeros rating count

df = df[df.ratings_count!= 0]

#Removing rows with zeros values

df.dropna(inplace=True)
#testinig data to see the change after cleaning 



df.isnull().sum()
df.info()
# Create a list to store the data

rating = []



for x in df['average_rating']: 

    if x >= 2.5 and x < 3.5:

        rating.append('Ok')

    elif x >= 3.5 and  x <3.9:

        rating.append('GOOD')

    elif x >= 3.9  and x < 4.2:

        rating.append('Very Good')

    elif x >= 4.2 :

        rating.append('Highly Recommended')

    else :

        rating.append('Disappointed')

# Create a column for the list

df['rating']= rating

#testing the change

df.head()
#creating a colum that contain the ratio of text review to rationg count

df['ratio']= df['text_reviews_count']/df['ratings_count']*100
df['ratio'].describe()
#Descriptive statistics for each numerical variables 

df.describe()
#distribution of num_pages,ratings_count, and text_reviews_count

np.seterr(divide = 'ignore')

# left plot: hist of ratings count

plt.figure(figsize = [12, 8])

plt.subplot(1, 3, 1)

log_data = np.log10(df['ratings_count']) # data transform

log_bin_edges = np.arange(0, log_data.max()+0.25,0.25)

plt.hist(log_data, bins = log_bin_edges)

plt.xlabel('log(ratings count)')



# central plot: hist of text reviews count

plt.subplot(1, 3, 2)

log_data = np.log10(df['text_reviews_count']) # direct data transform

log_bin_edges = np.arange(0, log_data.max()+0.25,0.25)

plt.hist(log_data, bins = log_bin_edges)

plt.xlabel('log(text reviews count)')



# right plot: # of pages 

plt.subplot(1, 3, 3)

plt.hist(df['  num_pages'], bins = 100)

plt.xlabel('Number of pages')

plt.xlim([50,1500]) #setting this limit because # of pages are in range lower than 1500

 
#average_rating distribution 

plt.hist(df['average_rating'],bins=60)

plt.xlabel('Average rating')

plt.ylabel('Count')

plt.xlim([1,5])
#Top 20 publishers

publishers = df.groupby('publisher')['bookID'].count().sort_values(ascending=False).head(20)

#plot the 20 top publisher based on the goodreads data 

#set the color 

base_color = sn.color_palette()[0]

# set the plot to the size of A4 paper

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

#plot

sn.barplot(publishers, publishers.index, color = base_color)

plt.title('Top 20 Publishers')

plt.xlabel('Counts')

plt.ylabel(' ');
#The most rated book

ratings_count=df.groupby('title')['ratings_count'].sum().sort_values(ascending=False).head(10)

# plot

fig, ax = plt.subplots()

fig.set_size_inches(10, 6)

sn.barplot(ratings_count, ratings_count.index, color="salmon")

plt.title('The most rated book')

plt.xlabel('Rating Counts')

plt.ylabel('-');
#Top 10 author

authors= df['authors'].value_counts().head(10)

#plot 

fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

sn.barplot(authors, authors.index,color = base_color)

plt.title('Authors')

plt.xlabel('Book Counts')

plt.ylabel('-');
df_numeric = ['average_rating' ,'  num_pages' ,'ratings_count','text_reviews_count','ratio']

sn.pairplot(df[df_numeric], diag_kind='kde');
#relationship between rating count and text reviews count

#sklearn.linear_model.LinearRegression

model = LinearRegression()

x = df['ratings_count']

y = df['text_reviews_count']

x= x.values.reshape(-1, 1)

y= y.values.reshape(-1, 1)

model.fit(x, y)

model = LinearRegression().fit(x, y)

y_pred = model.predict(x)

r_sq = model.score(x, y) 

r_sq #The Correlation Coefficient
# visualizing the relationship between rating count and text reviews count

# set the plot to the size of A4 paper

#plot

fig, ax = plt.subplots()

fig.set_size_inches(10, 8)

plt.scatter(x, y)

plt.plot(x, y_pred, color='red');

plt.title('Ratings Count Vs Text Reviews Count')

plt.xlabel('Ratings Count')

plt.ylabel('Text Reviews counts')

plt.xlim([0,3e6]);
#creat a subset with the 200 most rated books

df_highest = df.nlargest(200,['ratings_count'])

df_highest.head()
#creating two plots to show the distribution of rating count and text reviews count



# left plot: hist of ratings count

plt.figure(figsize = [12, 6])

plt.subplot(1, 2, 1)

sn.boxplot(x="rating", y="ratings_count", data=df_highest)

plt.ylim([0,2e6])

plt.xlabel('Rating')

plt.ylabel('Ratings Count')

plt.xticks(rotation=90,fontsize = 12);

# Right plot: hist of text reviews count

plt.subplot(1, 2, 2)

sn.boxplot(x="rating", y="text_reviews_count", data=df_highest)

plt.ylim([0,4e4])

plt.xlabel('Rating')

plt.ylabel('Text ReviwsCount')

plt.xticks(rotation=90,fontsize = 12);
df_publisher = df[df.groupby('publisher')['publisher'].transform('size') > 100]

df_publisher

plt.figure(figsize = [10, 8])

sn.violinplot(data = df_publisher, x = 'publisher', y = 'average_rating')

plt.xlabel(' ')

plt.ylabel('Average Rating')

plt.xticks(rotation=90,fontsize = 14);
#looking for authors with more than 25 books

df_author = df[df.groupby('authors')['authors'].transform('size') > 25]

df_author['authors'].unique()
# Creatinf dataframe for Stephen King books

df_king=df[df['authors']=='Stephen King']

df_king.rating.unique()
# Creatinf dataframe for P.G. Wodehouse books

df_Wodehouse=df[df['authors']=='P.G. Wodehouse']

df_Wodehouse.rating.unique()
# Creatinf dataframe for Agatha Christie books

df_Christie = df[df['authors']=='Agatha Christie']

df_Christie.rating.unique()
#creatin three plots for the selected authors 

#seting the size of the plots

plt.figure(figsize = [16, 8])

#first plot for Stephen King books

plt.subplot(1, 3, 1)

sn.kdeplot(df_king.ratio[df_king['rating'] == 'Ok'], shade=True, color="deeppink", label="Ok", alpha=.7)

sn.kdeplot(df_king.ratio[df_king['rating'] == 'Very Good'], shade=True, color="g", label="Very Good", alpha=.7)

sn.kdeplot(df_king.ratio[df_king['rating'] == 'GOOD'], shade=True, color="orange", label="Good", alpha=.7)

sn.kdeplot(df_king.ratio[df_king['rating'] == 'Highly Recommended'], shade=True, color="grey", label="Highly Recommended", alpha=.7)

plt.title('Stephen King')

#second plot : P.G. Wodehouse books

plt.subplot(1, 3, 2)

sn.kdeplot(df_Wodehouse.ratio[df_Wodehouse['rating'] == 'Very Good'], shade=True, color="g", label="Very Good", alpha=.7)

sn.kdeplot(df_Wodehouse.ratio[df_Wodehouse['rating'] == 'GOOD'], shade=True, color="orange", label="Good", alpha=.7)

sn.kdeplot(df_Wodehouse.ratio[df_Wodehouse['rating'] == 'Highly Recommended'], shade=True, color="grey", label="Highly Recommended", alpha=.7)

plt.title('P.G. Wodehouse')

#third plot : Agatha Christie

plt.subplot(1, 3, 3)

sn.kdeplot(df_Christie.ratio[df_Christie['rating'] == 'Ok'], shade=True, color="deeppink", label="Ok", alpha=.7)

sn.kdeplot(df_Christie.ratio[df_Christie['rating'] == 'Very Good'], shade=True, color="g", label="Very Good", alpha=.7)

sn.kdeplot(df_Christie.ratio[df_Christie['rating'] == 'GOOD'], shade=True, color="orange", label="Good", alpha=.7)

sn.kdeplot(df_Christie.ratio[df_Christie['rating'] == 'Highly Recommended'], shade=True, color="grey", label="Highly Recommended", alpha=.7)

plt.title('Agatha Christie')

df.groupby('authors')['bookID'].count().sort_values(ascending=False).head(10)



df_f = df[df['authors'].str.contains("Dostoyevsky")]

df_crime = df_f[df_f['title']== 'Crime and Punishment']

rating_crime=df_crime.groupby('publisher')['ratings_count'].sum().sort_values(ascending=False)

rating_crime
df_t = df[df['authors'].str.contains("Tolstoy")]

df_anna = df_t[df_t['title']== 'Anna Karenina']

rating_anna=df_anna.groupby('publisher')['ratings_count'].sum().sort_values(ascending=False)

rating_anna
