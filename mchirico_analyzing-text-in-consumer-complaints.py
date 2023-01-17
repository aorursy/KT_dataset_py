# Read in data from pandas
import pandas as pd

# This is used for fast string concatination
from io import StringIO

# Use nltk for valid words
import nltk
# Need to make hash 'dictionaries' from nltk for fast processing
import collections as co


import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Read the input
d = pd.read_csv("../input/consumer_complaints.csv") # the consumer dataset is now a Pandas DataFrame
# Only interested in data with consumer complaints
d=d[d['consumer_complaint_narrative'].notnull()]
# We want a very fast way to concat strings.
#  Try += if you don't believe this method is faster.
s=StringIO()
d['consumer_complaint_narrative'].apply(lambda x: s.write(x))

k=s.getvalue()
s.close()
k=k.lower()
k=k.split()
# Next only want valid strings
words = co.Counter(nltk.corpus.words.words())
stopWords =co.Counter( nltk.corpus.stopwords.words() )
k=[i for i in k if i in words and i not in stopWords]
s=" ".join(k)
c = co.Counter(k)

# Take a look at the 14 most common words
c.most_common(14)
s[0:100]
print(k[0:10],"\n\nLength of k %s" % len(k))
from wordcloud import WordCloud

# Read the whole text.
text = s

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt


# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(background_color="white",max_words=len(k),max_font_size=40, relative_scaling=.8).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
# Let's get some text involving identity theft
searchS='victim of identity theft'
vi = d[d['consumer_complaint_narrative'].str.find(searchS) >= 0]
d['victim']=None
d['e']=1
d['m']=None  # This will be for 'Closed with monetary relief'
d['victim'] = d[d['consumer_complaint_narrative'].str.find(searchS) >= 0]
d['m']=d[d['company_response_to_consumer'] == 'Closed with monetary relief']


# Take a look at some sample stories  mindex to mindex_inc
# Adjust this, to see different stories
mindex=20
mindex_inc=5+mindex
si=StringIO()
vi['consumer_complaint_narrative'].iloc[mindex:mindex_inc].apply(lambda x: si.write(x+'\n___\n\n'))

t=si.getvalue()
si.close()
print(t)
# We might be missing data on just fraud...
# Search for all cases of theft or fraud
searchS0='victim'
searchS1='identity'
searchS_OR=['theft','fraud']

vi2 = d[(d['consumer_complaint_narrative'].str.find(searchS0) >= 0) &
        (d['consumer_complaint_narrative'].str.find(searchS1) >= 0) &
       ( (d['consumer_complaint_narrative'].str.find(searchS_OR[0]) >= 0) |
        (d['consumer_complaint_narrative'].str.find(searchS_OR[1]) >= 0))
        ]
# vi2.count()

g=vi2.groupby(['issue'])
gg=g.count().reset_index()
gg.sort_values(by='e',inplace=True)
gg=g['e','victim','m'].count().reset_index()
gg.sort_values(by='e',inplace=True, ascending=False)
# Taking a look at common complaints
# Need to format this...but note only 9 cases where it
# was "Closed with monetary relief"  m==1

#gg.head(4)
with pd.option_context('display.max_rows', 10, 'display.max_columns', 4):
    print(gg)