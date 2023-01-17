# library imports

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# printing list of files available to us

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
bib = pd.read_csv("/kaggle/input/hindi-bible/Hindi_bible_with_authors.csv")

bib = bib.drop('Unnamed: 0', axis=1)

bib.head()
with open("/kaggle/input/hindi-bible/hindi_bible_books.txt",mode='r', encoding='utf-8-sig') as f:

    books = f.read()

books = books.split('\n')

NT_books = [i.strip('"') for i in books[39:66]]

print(NT_books)
df = pd.DataFrame(bib.groupby("Book Name").size()/(len(bib))*100, columns=["% occurrences"])

df["Testament"] = df.index.to_series().map(lambda x: 1 if x in NT_books else 0)

df = df.sort_values("% occurrences",ascending=False)

df.head()
from matplotlib.font_manager import FontProperties

import matplotlib.patches as mpatches

hindi_font = FontProperties(fname = "/kaggle/input/hindi-bible/Nirmala.ttf")

colors = {0:'red', 1:'blue'}

red_patch = mpatches.Patch(color='red',alpha=0.5, label='Old Testament')

blue_patch = mpatches.Patch(color='blue',alpha=0.5, label='New Testament')

plt.grid()

plt.bar(df.index, df["% occurrences"], align='center', alpha=0.5, color=df['Testament'].apply(lambda x: colors[x]))

plt.xticks(df.index, color="b", fontproperties=hindi_font, rotation=90, fontsize = 12)

plt.yticks(fontsize = 15)

plt.ylabel('% occurrences',fontsize = 20)

plt.title('Percentage Book wise portions',fontsize = 20)

plt.legend(handles=[red_patch,blue_patch])

plt.gca().margins(x=0)

plt.gcf().canvas.draw()

tl = plt.gca().get_xticklabels()

maxsize = max([t.get_window_extent().width for t in tl])

m = 0.5 # inch margin

s = maxsize/plt.gcf().dpi*55+2*m

margin = m/plt.gcf().get_size_inches()[0]



plt.gcf().subplots_adjust(left=margin, right=1.-margin)

plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
df = pd.DataFrame(bib.groupby("Authors").size()/(len(bib))*100, columns=["% occurrences"])

df = df.sort_values("% occurrences",ascending=False)

df.head()
plt.bar(df.index, df["% occurrences"], align='center', alpha=0.5)

plt.xticks(df.index, color="b", rotation=90, fontsize = 12)

plt.yticks(fontsize = 15)

plt.ylabel('% occurrences',fontsize = 15)

plt.title('Percentage portions of authors',fontsize = 15);

plt.gca().margins(x=0)

plt.gcf().canvas.draw()

tl = plt.gca().get_xticklabels()

maxsize = max([t.get_window_extent().width for t in tl])

m = 0.5 # inch margin

s = maxsize/plt.gcf().dpi*55+2*m

margin = m/plt.gcf().get_size_inches()[0]

plt.gcf().subplots_adjust(left=margin, right=1.-margin)

plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
set(bib[bib["Authors"]=="unknown"]["Book Name"])
df_O = pd.DataFrame(bib[bib['Testament Code']==0].groupby("Authors").size()/(len(bib[bib['Testament Code']==0]))*100, columns=["% occurrences"])

df_O = df_O.sort_values("% occurrences",ascending=False)

df_N = pd.DataFrame(bib[bib['Testament Code']==1].groupby("Authors").size()/(len(bib[bib['Testament Code']==1]))*100, columns=["% occurrences"])

df_N = df_N.sort_values("% occurrences",ascending=False)



f, axes = plt.subplots(1, 2,figsize=(13,4), gridspec_kw={'width_ratios': [3, 1]})

axes[0].bar(df_O.index, df_O["% occurrences"], align='center', alpha=0.5)

plt.sca(axes[0])

plt.xticks(df_O.index, color="b", rotation=90, fontsize = 12)

plt.title('Authors of Old Testament')

plt.ylabel('% occurences')



axes[1].bar(df_N.index, df_N["% occurrences"], align='center', alpha=0.5, color='r')

plt.sca(axes[1])

plt.xticks(df_N.index, color="b", rotation=90, fontsize = 12)

plt.title('Authors of New Testament')

plt.tight_layout();
with open("/kaggle/input/hindi-bible/Hindi_StopWords.txt",encoding='utf-8') as f:

    stopword= f.read().strip('\ufeff')

stopword = stopword.split(", ")

stopword = [i.strip("'") for i in stopword]

print(stopword)
with open("/kaggle/input/hindi-bible/Full_text_Bible.txt", mode='r', encoding='utf-8-sig') as f:

    text= f.read()

from wordcloud import WordCloud

from nltk.tokenize import word_tokenize

%matplotlib inline

stopwords = set(stopword)

wordcloud = WordCloud(font_path="/kaggle/input/hindi-bible/Nirmala.ttf",width = 800, height = 800, 

background_color ='white', 

stopwords = stopwords, 

min_font_size = 10).generate(text) 



# plot the WordCloud image 

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis('off') 

plt.tight_layout(pad = 0) 



plt.show()
def generate_stem_words(word):

    suffixes = {

    2: [u"कर",u"ाओ",u"िए",u"ाई",u"ाए",u"नी",u"ना",u"ते",u"ती",u"ाँ",u"ां",u"ों",u"ें"],

    3: [u"ाकर",u"ाइए",u"ाईं",u"ाया",u"ेगी",u"ेगा",u"ोगी",u"ोगे",u"ाने",u"ाना",u"ाते",u"ाती",u"ाता",u"तीं",u"ाओं",u"ाएं",u"ुओं",u"ुएं",u"ुआं"],    4: [u"ाएगी",u"ाएगा",u"ाओगी",u"ाओगे",u"एंगी",u"ेंगी",u"एंगे",u"ेंगे",u"ूंगी",u"ूंगा",u"ातीं",u"नाओं",u"नाएं",u"ताओं",u"ताएं",u"ियाँ",u"ियों",u"ियां"],

    5: [u"ाएंगी",u"ाएंगे",u"ाऊंगी",u"ाऊंगा",u"ाइयाँ",u"ाइयों",u"ाइयां"],

}

    for L in 5, 4, 3, 2:

        if len(word) > L + 1:

            for suf in suffixes[L]:

                if word.endswith(suf):

                    return word[:-L]

    return word
import collections

wordcount = {}

# To eliminate duplicates, we will split by punctuation, and use case demiliters.

for word in text.split():

    word = word.replace(".","")

    word = word.replace(",","")

    word = word.replace(":","")

    word = word.replace(";","")

    word = word.replace("\"","")

    word = word.replace("!","")

    word = generate_stem_words(word)

    if word not in stopwords:

        if word not in wordcount:

            wordcount[word] = 1

        else:

            wordcount[word] += 1

# most common word

word_counter = collections.Counter(wordcount)

freq_word={}

for word, count in word_counter.most_common(20):

    freq_word[word]=count

print(freq_word)

freq_df=pd.DataFrame(list(freq_word.items()), index=range(20), columns=['word', 'freq']) 

fig, ax = plt.subplots(figsize=(25,10))

ax.barh(freq_df['word'], freq_df['freq'], align='center')

ax.set_xlabel('Word frequencies', fontsize = 20)

ax.set_title('Top 20 most frequent words in Hindi Bible', fontsize = 20)

plt.yticks(range(len(freq_word.keys())),list(freq_word.keys()), fontproperties=hindi_font, fontsize = 20);
from string import punctuation

from nltk.probability import FreqDist

tokens = word_tokenize(text)

customStopWords = set(list(stopwords) + list(punctuation+'।'+'॥'))

wordsWOstopwords = [word for word in tokens if word not in customStopWords]

#removing numeric digits from list of words

wordsWOstopwords = [i for i in wordsWOstopwords if not i.isdigit()]

freq = FreqDist(wordsWOstopwords)



def freq_finder(word):

    """

    Input any Hindi word it will return how many times it appears in HHBD version of Bible.

    """

    return freq[word]

print("प्रेम appears for {} times while डर appears for {} times in HHBD Hindi Bible.".format(freq_finder('प्रेम'),freq_finder('डर')))
wrds = ['यीशु','मसीह','उद्धारकर्ता','उद्धार','क्रूस' ]

Ewrds = ['Jesus','Christ', 'Saviour','Salvation','Cross']

wrds_dict ={}

n=0

for i in wrds:

    wrds_dict[i+" "+"("+Ewrds[n]+")"]= freq_finder(i)

    n+=1

print(wrds_dict)
freq.pop('राजा', None)

freq
sents =[]

for i in text.split("॥"):

    sents.append(i.split('।'))

sents = [item for sublist in sents for item in sublist]
from collections import defaultdict

ranking = defaultdict(int)

for i,sent in enumerate(sents):

    for w in word_tokenize(sent):

        if w in freq:

            ranking[i] += freq[w]        
from heapq import nlargest

sents_indx = nlargest(1, ranking, key=ranking.get)

summary = [sents[j] for j in sorted(sents_indx)]

summary
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.hierarchy import ward, dendrogram

#define vectorizer parameters

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,

                                 min_df=0.2, stop_words=stopwords,

                                 use_idf=True, tokenizer=word_tokenize, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(sents[:10])



dist = 1 - cosine_similarity(tfidf_matrix)

linkage_matrix = ward(dist)

titles=['परमेश्वर','पृथ्वी','उजियाला','अन्धियारे','दिन','सांझ','पहिला','जल','ऊपर','आकाश']

fig, ax = plt.subplots(figsize=(6, 5)) # set size



ax = dendrogram(linkage_matrix, orientation="left", labels=titles);



plt.tick_params(\

    axis= 'x',          # changes apply to the x-axis

    which='both',      # both major and minor ticks are affected

    bottom='off',      # ticks along the bottom edge are off

    top='off',         # ticks along the top edge are off

    labelbottom='off')

plt.grid()

plt.yticks(fontproperties=hindi_font,fontsize = 15)

plt.tight_layout() #show plot with tight layout

plt.show();
colnames=["POS TAG","HWN ID","+ve score","-ve score","Related words"]

data = pd.read_csv("/kaggle/input/hindi-bible/HSWN_WN.txt", delimiter=' ',names=colnames,header=None)

data.head()
words_dict = {}

for i in data.index:

    words = data["Related words"][i].split(',')

    for word in words:

        words_dict[word] = (data["POS TAG"][i], data["+ve score"][i], data["-ve score"][i])

print("The size of the Hindi SentiWordNet: {} words".format(len(words_dict)))
from textblob import TextBlob

pos_data = pd.read_csv("/kaggle/input/hindi-bible/hindi word list.csv", header=None, names=["Hindi","English"])

pos_list = pos_data['English'].tolist()

pol_list=[]

for i in pos_list:

    blob =TextBlob(i)

    pol_list.append(blob.sentiment.polarity) 

pos_data["polarity"] = pol_list

pos_data.head()
senti_resource = set(list(words_dict.keys())+list(pos_data['Hindi']))

print("We have {} unique words without stopwords in Hindi Bible".format(len(set(wordsWOstopwords))))

print("And we have total {} unique words in our sentiment resources".format(len(senti_resource)))

remaining = [i for i in set(wordsWOstopwords) if i not in senti_resource]

print("The remaining words i.e. total unique words - (senti_resource): {} words".format(len(set(remaining))))
def sentiment(text):

    words = word_tokenize(text)

    words = [i for i in words if i not in customStopWords]

    pos_polarity = 0

    neg_polarity = 0

    #adverbs, nouns, adjective, verb are only used

    allowed_words = ['a','v','r','n']

    for word in words:

        if word in words_dict:

            #if word in dictionary, it picks up the positive and negative score of the word

            pos_tag, pos, neg = words_dict[word]

            if pos_tag in allowed_words:

                if pos > neg:

                    pos_polarity += pos

                elif neg > pos:

                    neg_polarity += neg

        elif word in pos_data['Hindi']:

            polarity = pos_data[pos_data['Hindi']== word]["polarity"]

            if polarity >= 0:

                pos_polarity += polarity

            elif polarity < 0:

                neg_polarity += polarity



    #calculating the no. of positive and negative words in total in a review to give class labels

    if pos_polarity > neg_polarity:

        return 1, pos_polarity

    else:

        return 0, -neg_polarity

print("Overall sentiment and it's polarity of statment: मैं इस उत्पाद से बहुत खुश हूँ is {}".format(sentiment("मैं इस उत्पाद से बहुत खुश हूँ")))
full_list = []

book_flag = range(66)

for j in book_flag:

    Chapter_txt=[]

    for i in bib.index:

        if bib["Book"][i]==book_flag[j]:

            Chapter_txt.append(bib["Text"][i])

    Chapter_str = "".join(Chapter_txt)

    full_list.append(Chapter_str)

print("Length of the resulting list: {}".format(len(full_list)))   
Books = [i.strip('"') for i in books[0:66]]

print(Books)
pol_list=[]

for i in full_list:

    polarity = sentiment(i)[1]

    pol_list.append(polarity)  

Polarity_dict = dict(zip(Books, pol_list))

pol_df = pd.DataFrame(

    {'Book': list(Polarity_dict.keys()),

     'Polarity': list(Polarity_dict.values())

    })

fig, ax = plt.subplots(figsize=(12,23))

ax.barh(pol_df["Book"],pol_df["Polarity"] , color='r');

ax.set_xlabel('Sentiment Scores', fontsize = 15)

ax.set_ylabel('Book Name', fontsize = 15)

ax.set_title('Cumulative Sentiment Score for each book', fontsize = 20)

plt.yticks(list(Polarity_dict.keys()), fontproperties=hindi_font, fontsize = 15);