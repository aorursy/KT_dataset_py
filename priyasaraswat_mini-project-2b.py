# Loading the required packages and libraries



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import nltk

from nltk import FreqDist

import warnings



import spacy

import gensim

from gensim import corpora



# Libraries for visualization

import seaborn as sns

import pyLDAvis

import pyLDAvis.gensim

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Input data files that are available

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df1 = pd.read_csv("/kaggle/input/airbnb-reviews_listing id 329997.csv")

df2 = pd.read_csv("/kaggle/input/airbnb-reviews_listing id 15440.csv")

df3 = pd.read_csv("/kaggle/input/airbnb-reviews_listing id 1260528.csv")
df1.head()
#Dropping columns that aren't required

df1 = df1.drop(['id','date','reviewer_id','reviewer_name'], axis=1)

df2 = df2.drop(['id','date','reviewer_id','reviewer_name'], axis=1)

df3 = df3.drop(['id','date','reviewer_id','reviewer_name'], axis=1)



#Dropping the rows with no comments

df1 = df1.dropna()

df2 = df2.dropna()

df3 = df3.dropna()
# Creating a new column to store the length of each review

df1['Comment_Length'] = df1['comments'].apply(len)

df2['Comment_Length'] = df2['comments'].apply(len)

df3['Comment_Length'] = df3['comments'].apply(len)
# Checking out the changes

df3.head()
#Getting descriptive statistics for the new column 'Comment_Length' 

df1.Comment_Length.describe()
df2.Comment_Length.describe()
df3.Comment_Length.describe()
# Identifying the location of the longest review

df1[df1['Comment_Length'] == 1598]['comments'].iloc[0]
df2[df2['Comment_Length'] == 2617]['comments'].iloc[0]
df3[df3['Comment_Length'] == 2359]['comments'].iloc[0]
# Using the plot style as seaborn-poster

plt.style.use('default')



# Plotting the histogram for the first data frame

df1['Comment_Length'].plot(bins=70, kind='hist') 
# Concatenated the three dataframe into one

df = pd.concat([df1,df2,df3])



# Visualize the comment length for all three listing using subplots 

plt.style.use('classic')

df.hist(column='Comment_Length', by='listing_id', bins=40,figsize=(12,4))
# Function to plot most frequent words

def freq_words(x, terms = 30):

    all_words = ' '.join([text for text in x])

    all_words = all_words.split()

    

    # using FreqDist function from nltk library to find number of

    # time a word has appeared

    fdist = FreqDist(all_words)

    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    

    # selecting top 10 most frequent words

    d = words_df.nlargest(columns="count", n = terms)

    sns.set_color_codes('pastel')

    plt.figure(figsize=(20,5))

    ax = sns.barplot(data=d, x= "word", y = "count")

    ax.set(ylabel = 'Count')

    plt.show()
freq_words(df1['comments'])
df1['comments'] = df1['comments'].str.replace("[^a-zA-Z#]", " ")
# Importing stop words list from the nltk.corpus

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
# Loading the pre-trained NLP model in spacy

nlp=spacy.load("en_core_web_lg") 





# Define a function to extract keywords/topics

def get_words(x):

    doc=nlp(x) ## Tokenize and extract grammatical components

    doc=[i.text for i in doc if i.text not in stop_words and i.pos_=="NOUN"] # Extracting nouns

    doc=list(map(lambda i: i.lower(),doc)) ## Normalize text to lower case

    doc=pd.Series(doc)

    doc=doc.value_counts().head().index.tolist() ## Get 5 most frequent nouns

    return doc
# Loading the extracted word from comments

key_words = []

for i in range(0, 501):

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if i == 395:

        continue

    else:

        key_words.append(get_words(df1['comments'][i]))



# Merging multiple lists into a single list

words = []

for x in  key_words:

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    for word in x:

        words.append(word)



print(words)
# Creating a new dataframe for the keywords and value count

df_new = pd.DataFrame()

df_new['Keywords'] = words



counts = df_new['Keywords'].value_counts()

df = pd.DataFrame(counts, df_new.Keywords.unique())

df = df.reset_index()

df = df.rename(columns={"index": "Topic", "Keywords": "Count"})



# Printing the top 15 words in the dataframe

df = df.sort_values('Count', ascending = False)

df.head(15)
plt.style.use('default')

freq_words(df_new['Keywords'])
dictionary = corpora.Dictionary(key_words)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in key_words]
# Creating the object for LDA model using gensim library

LDA = gensim.models.ldamodel.LdaModel



# Build LDA model

lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=7, random_state=100,

                chunksize=1000, passes=50)
lda_model.print_topics()
# Visualize the topics

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)

vis