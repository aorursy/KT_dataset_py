import pandas as pd

import numpy as np



pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

print("Setup Complete")
# Specify the path of the CSV file to read

my_filepath = "../input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv"



# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)
my_data.head()
my_data.info()
# Check out the overall trend in movie releases over the years around the world 

plt.figure(figsize=(10,6))

sns.distplot(a=my_data["Release Year"], kde=False)

plt.title("Number of movies realsed around the world \n over the years", loc="center")
my_data.rename(columns={"Origin/Ethnicity":"Origin"}, inplace=True)



# How many Origins are there in the dataset? 

len(my_data["Origin"].unique())
plt.figure(figsize=(10,5))

sns.catplot(x="Origin", kind="count", data=my_data, height=5, aspect=2)

plt.xticks(rotation=45, 

    horizontalalignment='right')

plt.title("Total number of movies released per ethnicity over the years (1900-2020)", fontsize=15)

plt.xlabel("")

plt.ylabel("")
equiv_dict = {"American":"The US", "Australian":"Australia", "Bangladeshi":"Bangladesh", 

              "British":"The Great Britain", "Canadian":"Canada", "Chinese":"China", 

              "Egyptian":"Egypt", "Hong Kong":"Hong Kong", "Fillipino":"The Phillipins", 

              "Assamese":"India", "Bengali":"India", "Bollywood":"India", "Kannada":"India", 

              "Malayalam":"India", "Marathi":"India", "Punjabi":"India", "Tamil":"India", 

              "Telugu":"India", "Japanese":"Japan", "Malaysian":"Malaysia", "Maldivian":"Maldives", 

              "Russian":"Russia", "South_Korean":"South_Korea","Turkish":"Turkey"}

my_data["Country"] = my_data["Origin"].map(equiv_dict)
plt.figure(figsize=(10,5))

sns.catplot(x="Country", kind="count", data=my_data, height=5, aspect=2)

plt.xticks(rotation=45, 

    horizontalalignment='right')

plt.title("Total number of movies released in each country over the years (1900-2020)", fontsize=15)

plt.xlabel("")

plt.ylabel("")
# Group the data by the "Country" and "Release Year" columns 

# to make visual the periods when the movie production was the most intensive for different countries.

by_country_by_year = my_data.groupby(["Country","Release Year"]).size().unstack()



plt.figure(figsize=(14,10))

g = sns.heatmap(

    by_country_by_year, 

    #square=True, # make cells square

    cbar_kws={'fraction' : 0.02}, # shrink colour bar

    cmap='OrRd', # use orange/red colour map

    linewidth=1 # space between cells

)
india = my_data[["Country", "Release Year"]].query('Country == "India" ').groupby("Release Year").size()



plt.figure(figsize=(10,5))

plt.title("Movie production industry growth in India")

plt.ylabel("Number of movies")

sns.lineplot(data=india)
# American word cloud



# Generate a word cloud image

usa = " ".join(plot for plot in my_data[my_data["Country"]=="The US"].Plot)

d = '../input/flags-pics2/'

usa_mask = np.array(Image.open(d + 'american-flag-1399556531Ci4.jpg'))

stopwords=set(STOPWORDS)

stopwords.update(["tell",'tells',"take","one","two","see","will","now"])

wordcloud_usa = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA", max_words=1000, mask=usa_mask).generate(usa)



# create coloring from image

image_colors = ImageColorGenerator(usa_mask)

plt.figure(figsize=[10,10])

plt.imshow(wordcloud_usa.recolor(color_func=image_colors), interpolation="bilinear")

plt.axis("off")



plt.show()
# Indian word cloud



# Generate a word cloud image

india = " ".join(plot for plot in my_data[my_data["Country"]=="India"].Plot)

d = '../input/flags-pics2/'

india_mask = np.array(Image.open(d + 'india-flag.jpg'))

stopwords=set(STOPWORDS)

stopwords.update(["tell",'tells',"take","one","two","see","will","now","meanwhile","give","ask"])

wordcloud_india = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA", max_words=1000, mask=india_mask).generate(india)



# create coloring from image

image_colors = ImageColorGenerator(india_mask)

plt.figure(figsize=[10,10])

plt.imshow(wordcloud_india.recolor(color_func=image_colors), interpolation="bilinear")

plt.axis("off")



plt.show()
pop_genres = list(my_data.Genre.unique())[:20]



plt.figure(figsize=(12,6))



sns.countplot(my_data.Genre,order=pd.value_counts(my_data.Genre).iloc[:20].index,palette=sns.color_palette("Pastel1", 20))

plt.title('Most frequent Genre types',fontsize=16)

plt.ylabel('Number of movies', fontsize=12)

plt.xlabel('Genre', fontsize=12)

plt.xticks(size=12,rotation=60)

plt.yticks(size=12)

sns.despine(bottom=True, left=True)

plt.show()
by_country_by_genre = my_data.groupby(["Country","Genre"]).size().unstack()

by_genre_top20 = by_country_by_genre.loc[:, by_country_by_genre.columns.isin(pop_genres)]



plt.figure(figsize=(14,8))

sns.heatmap(

    by_genre_top20, 

    #square=True, # make cells square

    cbar_kws={'fraction' : 0.02}, # shrink colour bar

    cmap='OrRd', # use orange/red colour map

    linewidth=1 # space between cells

)
# Getting rid of null values and invisible characters (non-breaking spaces)

top_cast = my_data[(my_data.Cast.notnull()) & (my_data.Cast != " ")]

top_cast.set_index("Cast",inplace=True)

top_cast.rename(index={'Three Stooges':'The Three Stooges'},inplace=True)
plt.figure(figsize=(14,10))

plt.title('Top cast (based on the number of movies)',fontsize=16)



sns.countplot(y=top_cast.index,order=pd.value_counts(top_cast.index)[:20].index,palette=sns.color_palette("Pastel1", 20))



plt.xlabel('Number of movies',fontsize=12)

plt.ylabel('',fontsize=12)

plt.yticks(size=12)

plt.show()
top_director = my_data[my_data.Director != "Unknown"]
plt.figure(figsize=(14,5))

plt.title('Top Directors (based on the number of movies directed)',fontsize=14)



sns.countplot(top_director.Director,order=pd.value_counts(top_director.Director)[:20].index,palette=sns.color_palette("Pastel1", 20))



plt.xlabel('',fontsize=10)

plt.ylabel('Number of movies directed',fontsize=10)

plt.xticks(size=11,rotation=60)

plt.show()