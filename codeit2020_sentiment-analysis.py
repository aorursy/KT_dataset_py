import pandas as pd
import numpy as np
from nltk import word_tokenize, RegexpTokenizer
from collections import Counter
from nltk.corpus import stopwords
from nltk import ngrams,FreqDist 
%matplotlib inline
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora
month=['Jan','feb','mar','apr','may','jun','july','aug','sep','oct','nov','dec']
fields=["Speech","month","year"]
#os.chdir('C:\\Users\Surabhi\\Desktop')
df=pd.DataFrame(pd.read_csv('Book1.csv',engine='python'))
row=df.shape
print(row)
print(df.columns)
df
#reviewing the start and ending of the files
#counting the word count of all the files
df['wc']=df['Speech_text'].apply(lambda x: len(x.split()))
df['wc'].describe()
df['unique']=df['month'].map(str)+df['year'].map(str)
#Representation of word count
#color=np.random.rand(36)
plt.figure(figsize=(10, 10))
plt.xticks(rotation=90)
plt.xlabel("Month and Year")
plt.ylabel("Word count")
plt.title("Word Counts of all Speeches")
plt.plot(df['unique'],df['wc'])
plt.grid()
plt.show()
df['Speech_text']=df['Speech_text'].str.lower()
tokenizer = RegexpTokenizer(r'\w+')
df['words_used']=df['Speech_text'].apply(lambda x:tokenizer.tokenize(x))
df['words_used'].head()
stop_words=set(stopwords.words('english'))
print(len(stop_words))
ls=["also","would","like","many","must","ki","us","get","even"]
for i in ls:
    stop_words.add(i)
stop_words
total_words=Counter()
for i in range(36):
    frequencies=Counter(df['words_used'][i])
    total_words+=frequencies
    print(i+1)
    for token,count in frequencies.most_common(40):
        if token not in stop_words and len(token)>2:
            print(token,count,sep="|")       
tok=[]
count_words=[]
for token,count in total_words.most_common(100):
    if token in stop_words:
        continue
    tok.append(token)
    count_words.append(count)
    print(token,count)
size=np.array(count_words)
color=np.random.rand(21)
plt.xticks(rotation=90)
plt.xlabel("Tokens")
plt.ylabel("Frequency")
plt.title("Most occuring words in Man Ki Baat")
plt.scatter(tok,count_words,s=size*2,c=color,alpha=0.8)
plt.grid(True)
plt.show()
corpus=' '.join(df['Speech_text'])
corpus=corpus.replace('.','. ')
wordcloud=WordCloud(background_color='white',stopwords=stop_words).generate(corpus)
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud)
#creating wordcloud here
#finding ngrams and their frequencies from the following phases
all_ngrams=dict()
count1=[]
tok1=[]
for i in range(36):
    for size in range (2,10):
        all_ngrams[size]=FreqDist(ngrams(df['words_used'][i],size))
print(all_ngrams[3].most_common(10))
tri_grams=Counter(all_ngrams[3])
for val,key in tri_grams.most_common(6):
    tok1.append(' '.join(val))
    count1.append(key)
colour=np.random.rand(6)
size=np.array(count)
plt.figure(figsize=(8, 10))
plt.xticks(rotation=90)
plt.xlabel("trigrams")
plt.ylabel("Frequency")
plt.title("Most occuring trigrams in Man Ki Baat")
plt.scatter(tok1,count1,s=size*10,c=colour,alpha=0.8,marker='h')
plt.grid(True)
plt.show()
#Deciding the corpus and generating a theme for it
#topic modelling
lemma=WordNetLemmatizer()
speech=[str(i) for i in df['Speech_text']]
stop_words.add("one")
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if len(ch)<2)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

data = [(clean(doc)).split() for doc in speech]

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(data)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in data]
doc_term_matrix
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=36, id2word = dictionary, passes=50)
print(ldamodel.print_topics(num_topics=37, num_words=3))
