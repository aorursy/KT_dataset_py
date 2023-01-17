import pandas as pd
import numpy as np
import seaborn as sbs
import matplotlib
import pandas_profiling as pandas_profiling


movies_data = pd.read_csv("1000 movies data.csv")     # Importing training dataset using pd.read_csv
profile = pandas_profiling.ProfileReport(movies_data)
profile.to_file(outputfile="movies_before_preprocessing.html")
movies_data.info()
#rename the columns to an easier name
movies_data.rename(columns={'Runtime (Minutes)': 'Runtime', 'Revenue (Millions)':'Revenue'}, inplace=True)

#Converting Year to datetime
movies_data['Year'] = pd.to_datetime(movies_data['Year'], format='%Y').dt.year

movies_data.info()
movies_data.describe(include = 'all')
movies_data[movies_data.duplicated(subset='Title')]
movies_data.loc[movies_data['Title'] == 'The Host']

#Split the genre into three columns
# new data frame with split value columns 
newGenre = movies_data["Genre"].str.split(",", expand = True)
newGenre.head()

movies_data['Genre1'] = newGenre[0]
movies_data['Genre2'] = newGenre[1]
movies_data['Genre3'] = newGenre[2]
movies_data.drop(labels='Genre',axis=1, inplace=True)
movies_data.head()
#Split the genre into three columns
# new data frame with split value columns 
actorsList = movies_data["Actors"].str.split(",", expand = True)
movies_data['Actor0'] = actorsList[0]
movies_data['Actor1'] = actorsList[1]
movies_data['Actor2'] = actorsList[2]
movies_data['Actor3'] = actorsList[3]
#movies_data.drop(labels='Actors',axis=1, inplace=True)
movies_data.head()
profile = pandas_profiling.ProfileReport(movies_data)
profile.to_file(outputfile="movies_after_preprocessing.html")
#get the actors of the top 100 movies in a Series
actors = (movies_data.sort_values(['Revenue'], ascending=False).head(50))['Actors']

actorsList = actors.tolist()
len(actorsList)

actorsFullList = []

for actors in actorsList:
    actorsList = actors.split(sep=',')
    for actor in actorsList:
        actorsFullList.append(actor.strip())

len(actorsFullList)

actors = pd.Series(actorsFullList)
actors.value_counts()
from wordcloud import WordCloud

# Convert the Series to string
text = actors.to_string(header=False, index=False)

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt


# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(width=1000, height=400, background_color="white", 
                              max_words= 150, max_font_size=50, relative_scaling=.5).generate(text)

plt.figure( figsize=(20,20) )
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

movies_data.plot.hexbin(x='Revenue', y='Runtime', gridsize=22)
import nltk
#nltk.download()
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')

sentence = "The Harrier sits at 1714 mm and the Safari at 1965 mm a difference of 251 mm. The rear section is raised almost by 50% of the wheel width which on 235/65R17 tyres is about 366 mm. This is true of the X2 platform as well on which the Safari is built. As the hump is to compensate the loss of height in the third row it needs to be at least 200 mm which makes it more or less a similar height as the Safari."
tokens = nltk.word_tokenize(sentence)
tokens

tagged = nltk.pos_tag(tokens)

entities = nltk.chunk.ne_chunk(tagged)
print(entities)

type(entities)

from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()

print([word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(sentence)) if pos[0] == 'N'])
#Fix metascore missing data
#Metascore has 64 / 6.4% missing values Missing
# I think this can be filled with the mean values assuming that it is an average movie

''' Median	59.5 AND Mean	58.985
So there is not much difference between the median and the mean. The distribution of the data also seems fairly symmetric
Using mean to replace missing values
'''

mean_metascore = movies_data.Metascore.mean()
print(mean_metascore)

print('Before fillna - ', movies_data['Metascore'].isnull().sum())
movies_data.Metascore.fillna(mean_metascore, inplace=True)
print('After fillna - ', movies_data['Metascore'].isnull().sum())
movies_data['Metascore'].plot.hist(bins=50)

#Fix revenue missing values
#Revenue has 128 / 12.8% missing values Missing
#Can be filled in with either the mean or the median checking on the spread of data. 

import seaborn as sns

mean_revenue = movies_data.Revenue.mean()
median_revenue = movies_data.Revenue.median()
print(mean_revenue, median_revenue)


'''
Interquartile range	100.45
Q1	13.27
Q3	113.72
Median	47.985
Mean	82.956

The distribution seems to be right skewed/positively skewed with the mean much greater than the median. 
Median is closer to Q1 than Q3
'''
#sns.barplot(movies_data['Revenue'])

#I am distributing the data randomly between the mean and the medium so as not to change the shape of the histogram drastically
#movies_data.Revenue.fillna(median_revenue, inplace=False).plot.hist(bins=40)
import random


print('Before fillna - ', movies_data['Revenue'].isnull().sum())
#movies_data.Revenue.fillna(median_revenue, inplace=True)
movies_data.Revenue.fillna(random.randint(int(median_revenue), int(mean_revenue)), inplace=False).plot.hist(bins=40)
print('After fillna - ', movies_data['Revenue'].isnull().sum())

movies_data['Revenue'].plot.hist(bins=40)

#Find out any correlation between genre and revenue

sns.boxplot(x="Genre1", y="Revenue", data=movies_data.sort_values('Revenue', ascending=False).head(100))
#How is the revenue distribution over the years?

movies_data.groupby(['Year'], as_index=False).sum().head()



#NOTE: As_index is very important, if year becomes index then it cannot be used in the plot
barplot = sns.barplot(x='Year', y='Revenue', data=movies_data.groupby(['Year'], as_index=False).sum())
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45)

#movies_data.groupby(['Year'], as_index=False).sum().plot.bar(x='Year', y='Revenue')
#Have scores become better or worse with the years? 
barplot = sns.barplot(x='Year', y='Metascore', data=movies_data.groupby(['Year'], as_index=False).sum())
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45)

#number of movies distribution over the years in the sample

#barplot = sns.barplot(x='Year', y='Count', data=movies_data.groupby(['Year'], as_index=False).sum())

movies_data.groupby(['Year'], as_index=False).count()

barplot = sns.barplot(x='Year', y='Rank', data=movies_data.groupby(['Year'], as_index=False).sum())
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45)


#Find out average score for every year
movies_data.groupby(['Year'], as_index=False).mean()

barplot = sns.barplot(x='Year', y='Rank', data=movies_data.groupby(['Year'], as_index=False).mean())
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45)
#Find out the unique genres

gen_unq = pd.concat([movies_data['Genre1'], movies_data['Genre2'], movies_data['Genre3']]).unique()
print(gen_unq)

movies_data.groupby(['Genre1','Genre2', 'Genre3'])['Revenue'].count()
boxplot = sns.boxplot(x="Genre1", y="Revenue",data=movies_data, palette="muted")
boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=45)

#ratings vs genre
#barplot = sns.barplot(x='Genre1', y='Rating', data = movies_data.groupby(['Genre1'], as_index=False))
barplot = sns.barplot(x='Genre1', y='Rating', data = movies_data.sort_values(['Rating']))
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45)
#Number of movies made per year
yearlyCount = movies_data['Year'].value_counts().sort_index().plot.bar()
yearlyCount.set_xticklabels(yearlyCount.get_xticklabels(), rotation=45)
movies_data.plot.scatter(x='Rating', y = 'Revenue')
movies_data.plot.scatter(x='Rating', y = 'Metascore')
movies_data.plot.scatter(x='Revenue', y = 'Runtime')



