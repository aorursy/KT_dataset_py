# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#count of popular words from IMDB dataset

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import nltk
top_Words = 10



df = pd.read_csv('../input/movie_metadata.csv',

                 usecols=['movie_title','plot_keywords', 'genres', 'title_year', 'director_name', 'imdb_score', 'num_critic_for_reviews'])

df.head()

df.duplicated().sum()

df[df.duplicated()]['movie_title']
group_by_year = df.groupby('title_year')

group_by_year['imdb_score'].mean().plot(kind='line', figsize=(8,4))

plt.title('Average IMDB score in Movies Titles by Year')

plt.ylabel('Average imdb score')

plt.xlabel('Year')
text = df.plot_keywords.str.lower().str.replace(r'\|', ' ').str.cat(sep=' ') #plot_keywords is a column name #str.cat(sep=''), sep refers to the string/none

words = nltk.tokenize.word_tokenize(text)

No_of_words = nltk.FreqDist(words) # FreqDist class is used to encode “frequency distributions”, which count the number of times of the word



stopwords = nltk.corpus.stopwords.words('english') #stopwords.words('english') returns a list of lowercase stop words.



words_without_stopwords = nltk.FreqDist(w for w in words if w not in stopwords) 

print('Including STOPWORDS:')

result = pd.DataFrame(No_of_words.most_common(top_Words), columns=['Word', 'Frequency'])

print(result)



print("=" * 50)



print('Excluding STOPWORDS:')

rslt = pd.DataFrame(words_without_stopwords.most_common(top_Words),columns=['Word', 'Frequency'])

print(rslt)



rslt = pd.DataFrame(words_without_stopwords.most_common(top_Words),columns=['Word', 'Frequency']).set_index('Word') #setting index , X-axis by  Word column



matplotlib.style.use('ggplot') #ggplot style is by default in matplotlib



rslt.plot.bar(rot=0) #rot=0 is to rotate the X-axis tick labels
import seaborn as sns

sns.set_style("whitegrid")



df_copy = df.copy().dropna()

director_critic_counts = df_copy.groupby(df_copy['director_name'])['num_critic_for_reviews'].sum()

director_critic_indx = director_critic_counts.sort_values(ascending=False)[:20].index

director_critic_values = director_critic_counts.sort_values(ascending=False)[:20].values



fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = director_critic_indx,

            y = director_critic_values,

            color='#90caf9',

            ax=ax)

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

plt.title('Director vs critic')

plt.ylabel('Counter')

plt.xlabel('Director')

del fig,ax,ticks
plt.hist(df.num_critic_for_reviews.dropna(), 50);

plt.title('Histogram of number of critics per movie')