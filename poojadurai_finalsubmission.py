#import libraries

import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
import gensim

from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
%matplotlib inline
#Read data
df=pd.read_csv("../input/unstructured-l0-nlp-hackathon/data.csv")
df.shape
df.head()
#Username extraction

#splittig the sentences into tokens
import re
from nltk.tokenize import word_tokenize , RegexpTokenizer
df['text']=df['text'].astype('str')
df['tokenized_text'] = df['text'].str.split() #create tokens

#picking up the user names present in each row
import re
mylist=df['text'].copy()
newlist=df['text'].copy()
for i in range(len(df)):
    mylist[i] = df['tokenized_text'][i]
    r = re.compile("^@.*") #Using regex to pick usernames
    newlist[i]=list(filter(r.match, mylist[i])) 
    
df['username']=newlist

#removing stopwords
import nltk
from nltk.corpus import stopwords

stop_words = nltk.corpus.stopwords.words('english')
for i in range(len(df)):
    for j in df['username'][i]:
        stop_words.append(j)
        
stop_words=[x.lower() for x in stop_words]
res=[]
for i in range(len(df)):
    if df['tokenized_text'][i]:
        my_list=df['tokenized_text'][i]
        my_list=[x.lower() for x in my_list] #convert to lower case
    

        words = [w for w in my_list if not w in stop_words]
        res.append(words)
    else:
         res.append(None)

df['nostopword']=res
df["nostopword"]= df["nostopword"].str.join(" ")

#remove url
df['nourl'] = df['nostopword'].replace(r'https\S+|http\S+|ftp\S+', ' ', regex=True)
df['nourl'][0:10]#no url in text


#remove numbers and punctuations
df["nopunct"] = df['nourl'].str.replace('[^\w\s]',' ')#removing punctuations

#cleaned text
df['cleaned_text'] = df['nopunct'].str.replace('\d+', ' ')
df['cleaned_text'].head()
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
#Preferred lemmatization over stemming
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

df["text_lemmatized"] = df["cleaned_text"].apply(lambda text: lemmatize_words(text))
df["text_lemmatized"].head()
#Found few odd(mis-spelt,short form,without space) words while going through data
df['text_lemmatized'] = df['text_lemmatized'].str.replace('bdrm','bedroom')
df['text_lemmatized'] = df['text_lemmatized'].str.replace('teslas','tesla')
df['text_lemmatized'] = df['text_lemmatized'].str.replace('wattached','attached')
df['text_lemmatized'] = df['text_lemmatized'].str.replace('quartermiles','quarter miles')
df['text_lemmatized'] = df['text_lemmatized'].str.replace('friendlyquiet','friendly quiet')
df['text_lemmatized'] = df['text_lemmatized'].str.replace('jets','jet')
df['text_lemmatized'] = df['text_lemmatized'].str.replace('sq','sqft')
df['text_lemmatized'] = df['text_lemmatized'].str.replace('ft','sqft')
#Getting frequency of all words

a = df['text_lemmatized'].str.lower().str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(a)
word_dist = nltk.FreqDist(words)
print (word_dist)

count = pd.DataFrame(word_dist.most_common(252082),
                    columns=['Word', 'Frequency'])
count.head()
#checking how many words occur only once
count[count['Frequency']<2].shape
#Filtering words with threshold less than 2

stop=count[count['Frequency']<2]
stop.reset_index(inplace=True)
#Adding those filtered words to stop words list

stop_words1=stop_words.copy()
for i in range(len(stop)):
        stop_words1.append(stop['Word'][i])
print(stop_words1)
#Getting cleaned text without less frequency words(Threshold is less than 2)

df['tokens_clean'] = df['text_lemmatized'].str.split()
res=[]
for i in range(len(df)):
    if df['tokens_clean'][i]:
        my_list=df['tokens_clean'][i]
        my_list=[x.lower() for x in my_list] #convert to lower case
    

        words = [w for w in my_list if not w in stop_words1]
        res.append(words)
    else:
         res.append(None)

df['new_text']=res
df["new_text"]= df["new_text"].str.join(" ") #contains words with  frequency>2

#Applying pos filter

#finding pos tags for each row
df['new_text']=df['new_text'].astype('str')
aa=df['new_text']
r=df['new_text'].copy()
for i in range(len(df)):
    r[i] = nltk.pos_tag(word_tokenize(df['new_text'][i]))
df['pos_tags']=r
#Going to eliminate words with tags CC,CD,MD.TO,IN
df.insert(loc=2, column='buffer', value=['' for i in range(df.shape[0])])
a1=df['buffer'].copy()
for i in range(len(df)):
    for j in range(len(df['pos_tags'][i])):
        if (df['pos_tags'][i][j][1]=='CC') | (df['pos_tags'][i][j][1]=='CD') | (df['pos_tags'][i][j][1]=='MD') | (df['pos_tags'][i][j][1]=='TO') | (df['pos_tags'][i][j][1]=='IN'):
             a1[i]=a1[i]+' '+(df['pos_tags'][i][j][0])
lis=df['buffer'].copy()
for i in range(len(df)):
    lis[i]=list(a1[i].split(" ")) 
    lis[i]=lis[i][1:]
lis2=[]
for i in range(len(df)):
    lis2.extend(lis[i])
stopp=list(set(lis2))
len(stopp)
#Removing the mentioned pos tag words from cleaned text

res=[]
df['new_text'] = df['new_text'].str.split()
for i in range(len(df)):
    if df['new_text'][i]:
        my_list=df['new_text'][i]
        words = [w for w in my_list if not w in stopp]
        res.append(words)
    else:
         res.append(None)
df['new_text_posfilter']=res
df["new_text_posfilter"]= df["new_text_posfilter"].str.join(" ")

#Building  lda model with the cleaned text that has been subjected to pos tag filter and frequency filter

data_text_1 = pd.DataFrame(df['new_text_posfilter'])
data_text_1 = data_text_1.astype('str')
for idx in range(len(data_text_1)): 
    #go through each word in each data_text row and set them on the index.
    data_text_1.iloc[idx]['new_text_posfilter'] = [word for word in data_text_1.iloc[idx]['new_text_posfilter'].split(' ')];
#get the words as an array for lda input

train_1= [value[0] for value in data_text_1.iloc[0:].values]

import gensim.corpora as corpora

import time 
dictionary = gensim.corpora.Dictionary(train_1)
bow = [dictionary.doc2bow(line) for line in train_1]
#eta tracks how words are allocated to terms
#When not provided, or provided as the keyword 'auto', gensim presupposes an even distribution across terms and topics
#First I am keeping 'auto' for eta
#I'll first train a topic model on the corpus of sentences  using the 'auto' keyword.

ww=[]
def test_eta(eta, dictionary, ntopics, print_topics=True, print_dist=True):
    np.random.seed(42) # set the random seed for repeatability
    bow = [dictionary.doc2bow(line) for line in train_1] # get the bow-format lines with the set dictionary
    with (np.errstate(divide='ignore')):  # ignore divide-by-zero warnings
        model = gensim.models.ldamodel.LdaModel(
            corpus=bow, id2word=dictionary, num_topics=ntopics,
            random_state=42, chunksize=100, eta=eta,
            eval_every=-1, update_every=1,
            passes=150, alpha='auto', per_word_topics=True)
    
    print('Perplexity: {:.2f}'.format(model.log_perplexity(bow)))
    if print_topics:
        # display the top 30 terms for each topic
        for topic in range(ntopics):
            print('Topic {}: {}'.format(topic, [dictionary[w] for w,p in model.get_topic_terms(topic, topn=30)]))
    if print_dist:
        # display the topic probabilities for each document
        for line,bag in zip(txt,bow):
            doc_topics = ['({}, {:.1%})'.format(topic, prob) for topic,prob in model.get_document_topics(bag)]
            print('{} {}'.format(line, doc_topics))
            zipp=[[topic, prob] for topic,prob in model.get_document_topics(bag)]
            ww.append(zipp)
    return ww,model
%%time
txt=df['text'].to_list()
ww,model=test_eta('auto',dictionary,ntopics=5)
#Coherence score

from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda_1 = CoherenceModel(model=model, texts=train_1, dictionary=dictionary, coherence='c_v')
coherence_lda_1 = coherence_model_lda_1.get_coherence()
print('\nCoherence Score: ', coherence_lda_1)
#Topic 0 word cloud
topic0=[dictionary[w] for w,p in model.get_topic_terms(0, topn=50)]
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline
pf=pd.DataFrame(topic0,columns=['word'])

al = ' '.join(pf['word'].str.lower())

wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=1000000).generate(al)
rcParams['figure.figsize'] = 100, 200
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
#Topic 0 looks something on tecnology,business and development.So I am assuming this to be tech_news
#Topic 1 word cloud
topic0=[dictionary[w] for w,p in model.get_topic_terms(1, topn=50)]
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline
pf=pd.DataFrame(topic0,columns=['word'])

al= ' '.join(pf['word'].str.lower())

wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=1000000).generate(al)
rcParams['figure.figsize'] = 100, 200
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
#Topic 1 mainly revolves on rooms,home,so I am assuming this to be room_rentals
#Topic 2 word cloud
topic0=[dictionary[w] for w,p in model.get_topic_terms(2, topn=50)]
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline
pf=pd.DataFrame(topic0,columns=['word'])

al = ' '.join(pf['word'].str.lower())

wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=1000000).generate(al)
rcParams['figure.figsize'] = 100, 200
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
#Topic 2 revolves around match,goal.I am assuming this to be sports_news
#Topic 3 word cloud
topic0=[dictionary[w] for w,p in model.get_topic_terms(3, topn=50)]
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline
pf=pd.DataFrame(topic0,columns=['word'])

al = ' '.join(pf['word'].str.lower())

wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=1000000).generate(al)
rcParams['figure.figsize'] = 100, 200
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
#Topic 3 revolves around work,employee,pros and cons.I am assuming this to be glassdoor_reviews
#Topic 4 word cloud
topic0=[dictionary[w] for w,p in model.get_topic_terms(4, topn=50)]
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline
pf=pd.DataFrame(topic0,columns=['word'])

al = ' '.join(pf['word'].str.lower())

wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=1000000).generate(al)
rcParams['figure.figsize'] = 100, 200
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
#Topic 4 revolves around car and drive.I am assuming this to be automobile
# Visualize the topics
import pyLDAvis
import pyLDAvis.gensim 
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model, bow, dictionary)
vis

#This is not bad but I am going to use seeded words which are priori words to see how it goes
#Now that we know what is the theme of each topic,we can add seeded words for each topic, just a few, that helps out  to push the model in a certain direction.
#The accuracy score is 0.89135
#eta tracks how words are allocated to terms
#When not provided, or provided as the keyword 'auto', gensim presupposes an even distribution across terms and topics
#First I am keeping 'auto' for eta
#I'll first train a topic model on the corpus of sentences  using the 'auto' keyword.

ww1=[]
def test_eta1(eta, dictionary, ntopics, print_topics=True, print_dist=True):
    np.random.seed(42) # set the random seed for repeatability
    bow = [dictionary.doc2bow(line) for line in train_1] # get the bow-format lines with the set dictionary
    with (np.errstate(divide='ignore')):  # ignore divide-by-zero warnings
        model = gensim.models.ldamodel.LdaModel(
            corpus=bow, id2word=dictionary, num_topics=ntopics,
            random_state=42, chunksize=100, eta=eta,
            eval_every=-1, update_every=1,
            passes=150, alpha='auto', per_word_topics=True)
    
    print('Perplexity: {:.2f}'.format(model.log_perplexity(bow)))
    if print_topics:
        # display the top 30 terms for each topic
        for topic in range(ntopics):
            print('Topic {}: {}'.format(topic, [dictionary[w] for w,p in model.get_topic_terms(topic, topn=30)]))
    if print_dist:
        # display the topic probabilities for each document
        for line,bag in zip(txt,bow):
            doc_topics = ['({}, {:.1%})'.format(topic, prob) for topic,prob in model.get_document_topics(bag)]
            print('{} {}'.format(line, doc_topics))
            zipp=[[topic, prob] for topic,prob in model.get_document_topics(bag)]
            ww1.append(zipp)
    return ww1,model
#To define a prior distribution, we need to create a numpy matrix with the same number of rows and columns as topics and terms, respectively. 
#We then populate that matrix with our prior distribution. To do this we pre-populate all the matrix elements with 1, then with a really high number for 
#the elements that correspond to  guided term-topic distribution.
def create_eta(priors, etadict, ntopics):
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=1) # create a (ntopics, nterms) matrix and fill with 1
    for word, topic in priors.items(): # for each word in the list of priors
        keyindex = [index for index,term in etadict.items() if term==word] # look up the word in the dictionary
        if (len(keyindex)>0): # if it's in the dictionary
            eta[topic,keyindex[0]] = 1e7  # put a large number in there
    eta = np.divide(eta, eta.sum(axis=0)) # normalize so that the probabilities sum to 1 over all topics
    return eta
#Since I have identified the topics,its easy for me to put related words under each topic number

apriori_original = {
    'tech':0, 'technology':0,'science':0,'tesla':0, 'elon':0, 'musk':0,'energy':0,'space':0,'orbit':0,'microsoft':0,'virtual':0,'reality':0,
    'bedroom':1, 'bathroom':1,'suite':1, 'rent':1,'kitchen':1,'resort':1,'home':1,
    'hockey':2,'match':2,'lead':2,'player':2,'team':2,'game':2,'cup':2,'victory':2,'champion':2,'league':2,'madrid':2,'barcelona':2,
    'pro':3, 'con':3,'hr':3, 'policy':3,
     
    'car':4, 'mile':4, 'insurance':4,'driver':4,'sedan':4,'jet':4
    
    
}
eta = create_eta(apriori_original, dictionary, 5)
ww1,model=test_eta1(eta, dictionary, 5)
#Coherence score

from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda_1 = CoherenceModel(model=model, texts=train_1, dictionary=dictionary, coherence='c_v')
coherence_lda_1 = coherence_model_lda_1.get_coherence()
print('\nCoherence Score: ', coherence_lda_1)
#The coherence score has improved from 0.48 to 0.50
#This model has improved the accuracy score from 0.89135 to 0.90676
#finding out the maximum probability topic number for each Id
a=[]

for i in range(len(ww1)):
    list1=[]
    list2=[]
    for j in range(len(ww1[i])):
        list1.append(ww1[i][j][0])
        
        list2.append(ww1[i][j][1])
       
    maxind=pd.Series(list2).idxmax()
    
    a.append(list1[maxind])
len(ww1)
count=[]
#assigning topic names
for i in range(len(a)):
    if a[i]==0:
        count.append("tech_news")
    elif a[i]==1:
        count.append("room_rentals")
    elif a[i]==2:
        count.append("sports_news")
    elif a[i]==3:
        count.append("glassdoor_reviews")
    else:
        count.append("Automobiles")
df['topic']=count
bf=df[['Id','topic']]
bf.shape















