# for some basic operations
import numpy as np
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# for providing path
import os
print(os.listdir('../input/'))
# reading the data

data = pd.read_csv('../input/job_skills.csv')

# getting the shape of the data
data.shape
# checking the head of the data

data.head()
# describing the data set

data.describe()
# cheking the null values in the dataset

data.isnull().any()
# The Companies involved

plt.rcParams['figure.figsize'] = (7, 9)
plt.style.use('_classic_test')

data['Company'].value_counts().plot.bar(color = 'cyan')
plt.title('The companies Involved for recruitment', fontsize = 20)
plt.xlabel('Names of companies', fontsize = 15)
plt.show()
# checking most popular top 20 types of job Titles 

plt.rcParams['figure.figsize'] = (19, 8)

color = plt.cm.PuRd(np.linspace(0, 1, 20))
data['Title'].value_counts().sort_values(ascending = False).head(20).plot.bar(color = color)
plt.title("Most Popular 20 Job Titles of Google", fontsize = 20)
plt.xlabel('Names of Job Titles', fontsize = 18)
plt.ylabel('count', fontsize = 15)
plt.show()
# checking most popular top 20 types of Job Categories

plt.rcParams['figure.figsize'] = (19, 8)

color = plt.cm.BuPu(np.linspace(0, 1, 20))
data['Category'].value_counts().sort_values(ascending = False).head(20).plot.bar(color = color)
plt.title("Most Popular 20 Job Categories of Google", fontsize = 20)
plt.xlabel('Names of Job Categories', fontsize = 15)
plt.ylabel('count', fontsize = 15)
plt.show()
# checking most popular top 20 types of job Locations

plt.rcParams['figure.figsize'] = (19, 8)

color = plt.cm.hsv(np.linspace(0, 1, 20))
data['Location'].value_counts().sort_values(ascending = False).head(20).plot.bar(color = color)
plt.title("Most Popular 20 Job Locations of Google", fontsize = 20)
plt.xlabel('Names of Job Locations', fontsize = 15)
plt.ylabel('count', fontsize = 15)
plt.show()
# checking most popular job destinations

from wordcloud import WordCloud
from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'green',
                      width = 2000,
                      height = 2000).generate(str(data['Location']))

plt.rcParams['figure.figsize'] = (12, 12)
plt.axis('off')
plt.imshow(wordcloud)
plt.title('Location with Most Jobs', fontsize = 30)
plt.show()
# checking the most popular Responsibilities

from wordcloud import WordCloud
from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'lightgreen',
                      max_words = 100,
                      width = 2000,
                      height = 2000).generate(str(data['Responsibilities']))

plt.rcParams['figure.figsize'] = (12, 12)
plt.axis('off')
plt.imshow(wordcloud)
plt.title('Most Available Responsibilities', fontsize = 30)
plt.show()
# checking the most popular Minimum edu. requirements

from wordcloud import WordCloud
from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'lightgreen',
                      max_words = 80,
                      width = 2000,
                      height = 2000).generate(str(data['Minimum Qualifications']))

plt.rcParams['figure.figsize'] = (12, 12)
plt.axis('off')
plt.imshow(wordcloud)
plt.title('Most Required Educational Requirements', fontsize = 30)
plt.show()
# checking the most popular Minimum edu. requirements

from wordcloud import WordCloud
from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'yellow',
                      max_words = 80,
                      width = 2000,
                      height = 2000).generate(str(data['Preferred Qualifications']))

plt.rcParams['figure.figsize'] = (12, 12)
plt.axis('off')
plt.imshow(wordcloud)
plt.title('Most Demanded Preferred Requirements', fontsize = 30)
plt.show()