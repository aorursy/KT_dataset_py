# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import nltk
from string import punctuation
# Any results you write to the current directory are saved as output.
sent="India is a republic country. We are proud Indians"
text1=nltk.word_tokenize(sent)
vocab=sorted(set(text1))
print(vocab)
# print(punctuation(vocab))
print(list(punctuation))
vocab1=[]
for i in vocab:
    if i not in punctuation:
        vocab1.append(i)
print(vocab1)
from nltk.stem.snowball import SnowballStemmer
stemmed_vocab=[]
stemObj = SnowballStemmer("english")
for i in vocab1:
    stemmed_vocab.append(stemObj.stem(i))
print(stemmed_vocab)
from nltk.stem.wordnet import WordNetLemmatizer
lemmaObj =  WordNetLemmatizer()
lemmaObj.lemmatize("went",pos='v')
# lemmaObj =  WordNetLemmatizer()
# pos_list=worpos_tag(vocab1)
# print(pos_list)
# lemma_vocab=[]
# for i,pos in pos_list:
#     lemma_vocab.append(lemmaObj.lemmatize(i,pos=pos))
# print(lemma_vocab)
from nltk.corpus import stopwords
wo_stop_words=[]
stop_words=set(stopwords.words("english"))
for i in vocab1:
    if i not in stop_words:
        wo_stop_words.append(i)
print(wo_stop_words)
vocab_wo_punct=vocab1
from nltk import pos_tag
pos_list = pos_tag(vocab_wo_punct)
print(pos_list)

vocab_no_punct=vocab1
from nltk import FreqDist
FreqDist(vocab_no_punct).plot()
from nltk import ngrams
from nltk import ConditionalFreqDist
#use 2 for bigrams
bigrams = ngrams(vocab_no_punct,2)
print(list(bigrams))
nltk.ConditionalFreqDist(list(bigrams))
# from nltk.corpus import brown
# cfd = nltk.ConditionalFreqDist(
#           (genre, word)
#            for genre in brown.categories()
#            for word in brown.words(categories=genre))
# cfd.tabulate().shape
# t1 = "This is Anand here"
# t2 = "we have now breached the inner circle"
# t3 = "I dont think i want to not wish you a merry christmas"
# text = [t1,t2,t3]
# # words = ["we","the"]`a`
# nltk.ConditionalFreqDist()
from nltk import ngrams
#use 2 for trigrams
trigrams = ngrams(vocab_no_punct,3)
print(list(trigrams))
from nltk import ngrams
#use 2 for bigrams
fourgrams = ngrams(vocab_no_punct,4)
print(list(fourgrams))
text="I saw John coming. He was with Mary. I talked to John and Mary. \
John said he met Mary on the way. John and Mary were going to school."

nltk.FreqDist(nltk.word_tokenize(text)).plot()
from nltk.book import *

from nltk import Text
vocab_no_punct.append("India")
text_nltk = Text(vocab_no_punct)
type(text_nltk)
text_nltk.dispersion_plot(['India', 'Indians', 'We', 'a', 'are','country', 'is', 'proud', 'republic'])
import nltk
import string
from nltk import word_tokenize
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk import ngrams
file = open("../input/nlp-text-sample/nlp wikipedia sample.txt",'r')
# lst
text = ''
for i in file.readlines():
    text += i
# print(text)
trimmed_text = text.strip()
# print(trimmed_text)
converted_text = trimmed_text.lower()
# print(converted_text)
tokenized_list = word_tokenize(converted_text)
# print(tokenized_list)
punct_tokenized_list = wordpunct_tokenize(converted_text)
# print(punct_tokenized_list)
vocab_set = set(tokenized_list)
# print(vocab_list)
set_wo_stopwords = vocab_set - set(stopwords.words("english"))
# print(set_wo_stopwords)
set_wo_punctuation = set_wo_stopwords - set(punctuation)
# print(set_wo_punctuation)
stemmed_list= []
stemObj = SnowballStemmer("english")
for i in set_wo_punctuation:
    stemmed_list.append(stemObj.stem(i))
# print(stemmed_list)
pos_tag_list = pos_tag(set_wo_punctuation)
print(pos_tag_list)
def parts_of_speech(pos):
    if pos.startswith("N"):
        return wordnet.NOUN
    elif pos.startswith("J"):
        return wordnet.ADJ
    elif pos.startswith("V"):
        return wordnet.VERB
    elif pos.startswith("R"):
        return wordnet.ADV
    elif pos.startswith("S"):
        return wordnet.ADJ_SAT
    else:
        return ''
lemma_list = []
lemmaObj =  WordNetLemmatizer()
for word,pos in pos_tag_list:
    get_pos = parts_of_speech(pos)
    if get_pos != '':
        lemma_list.append(lemmaObj.lemmatize(word, pos = get_pos))
    else:
        lemma_list.append(word)
# print(lemma_list)
bigrams = ngrams(set_wo_punctuation,2)
# print(list(bigrams))
from nltk import RegexpParser
grammar = r"""NP: {<DT|PP\$>?<JJ>*<NN>}
           VP: {<NNP>+}"""
regParser = RegexpParser(grammar)
#  print(regParser.parse(pos_tag(punct_tokenized_list)))
barack = """Barack Hussein Obama II born August 4, 1961) is an American politician who served as the 44th President of 
the United States from January 20, 2009, to January 20, 2017. A member of the Democratic Party, he was the 
first African American to assume the presidency and previously
served as a United States Senator from Illinois (2005–2008)."""

from nltk import ne_chunk
barack = """Barack Hussein Obama II born August 4, 1961) is an American politician
who served as the 44th President of 
the United States from January 20, 2009, to January 20, 2017.
A member of the Democratic Party, he was the 
first African American to assume the presidency and previously
served as a United States Senator from Illinois (2005–2008)."""
tokenised_barack = word_tokenize(barack)
pos_list = pos_tag(tokenised_barack)
ne_ch = ne_chunk(pos_list)
# ne_ch.draw()
print(ne_ch)
from nltk import RegexpParser
from nltk import word_tokenize
from nltk import pos_tag

barack = """Barack Hussein Obama II born August 4, 1961) is an American politician
who served as the 44th President of 
the United States from January 20, 2009, to January 20, 2017.
A member of the Democratic Party, he was the 
first African American to assume the presidency and previously
served as a United States Senator from Illinois (2005–2008)."""

grammar = r"""Place: {<NNP><NNPS>+}
           Date: {<NNP><CD><,><CD>}
           Person: {<NNP>+}
           """

tokenised_barack = word_tokenize(barack)
pos_list = pos_tag(tokenised_barack)
regParser = RegexpParser(grammar)
reg_lines = regParser.parse(pos_list)
# print(reg_lines)
#default tagger
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import brown
from nltk.tag import DefaultTagger

tagged_sentences = brown.tagged_words(categories = "news") 

tags = [tag for word,tag in tagged_sentences]

barack = """Barack Hussein Obama II born August 4, 1961) is an American politician
who served as the 44th President of 
the United States from January 20, 2009, to January 20, 2017.
A member of the Democratic Party, he was the 
first African American to assume the presidency and previously
served as a United States Senator from Illinois (2005–2008)."""
tokenised_barack = word_tokenize(barack)
default_tag = nltk.FreqDist(tags).max()
default_tagger = nltk.DefaultTagger(default_tag)
tagged_barack = default_tagger.tag(tokenised_barack)
# print(tagged_barack)
# print(default_tagger.evaluate([pos_list]))
#regexp tagger
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import brown
from nltk.tag import RegexpTagger

patterns = [
     (r'.*ing$', 'VBG'),               # gerunds
     (r'.*ed$', 'VBD'),                # simple past
     (r'.*es$', 'VBZ'),                # 3rd singular present
     (r'.*ould$', 'MD'),               # modals
     (r'.*\'s$', 'NN$'),               # possessive nouns
     (r'.*s$', 'NNS'),                 # plural nouns
     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'.*', 'NN')                     # nouns (default)
]

barack = """Barack Hussein Obama II born August 4, 1961) is an American politician
who served as the 44th President of 
the United States from January 20, 2009, to January 20, 2017.
A member of the Democratic Party, he was the 
first African American to assume the presidency and previously
served as a United States Senator from Illinois (2005–2008)."""

tokenised_barack = word_tokenize(barack)
regexp_tagger = nltk.RegexpTagger(patterns)
tagged_barack = regexp_tagger.tag(tokenised_barack)
# print(tagged_barack)
# regexp_tagger.evaluate([pos_list])

#lookup tagger
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import brown
from nltk.tag import UnigramTagger

fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, tag) in most_freq_words)


barack = """Barack Hussein Obama II born August 4, 1961) is an American politician
who served as the 44th President of 
the United States from January 20, 2009, to January 20, 2017.
A member of the Democratic Party, he was the 
first African American to assume the presidency and previously
served as a United States Senator from Illinois (2005–2008)."""

tokenised_barack = word_tokenize(barack)

unigram_tagger = nltk.UnigramTagger(model=likely_tags)
tagged_barack = unigram_tagger.tag(tokenised_barack)
# print(tagged_barack)
# unigram_tagger.evaluate([pos_list])
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import brown
from nltk.tag import UnigramTagger

brown_tagged_sents = brown.tagged_sents(categories='news')

barack = """Barack Hussein Obama II born August 4, 1961) is an American politician
who served as the 44th President of 
the United States from January 20, 2009, to January 20, 2017.
A member of the Democratic Party, he was the 
first African American to assume the presidency and previously
served as a United States Senator from Illinois (2005–2008)."""

tokenised_barack = word_tokenize(barack)

print(brown_tagged_sents)

unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
tagged_barack = unigram_tagger.tag(tokenised_barack)
# print(tagged_barack)
# print(unigram_tagger.evaluate([pos_list]))
# training_list = 
#n-gram tagging
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import brown
from nltk.tag import BigramTagger

brown_tagged_sents = brown.tagged_sents(categories='news')

barack = """Barack Hussein Obama II born August 4, 1961) is an American politician
who served as the 44th President of 
the United States from January 20, 2009, to January 20, 2017.
A member of the Democratic Party, he was the 
first African American to assume the presidency and previously
served as a United States Senator from Illinois (2005–2008). President of the United States Donald John Trump (born June 14, 1946) succeeded him"""

trump ="""President of the United States Donald John Trump (born June 14, 1946) is the 45th President of the United States. 
Before entering politics, 
he was a businessman and television personality."""

trump_pos_list = pos_tag(word_tokenize(trump))

# print(trump_pos_list)

tokenised_barack = word_tokenize(barack)
bigrams = ngrams(tokenised_barack,2)
bigram_tagger = nltk.NgramTagger(n=3,train=[pos_list])
tagged_trump = bigram_tagger.tag(word_tokenize(barack))
print(tagged_trump)
print(bigram_tagger.evaluate([trump_pos_list]))
# pos_list
words = [w for w,_ in brown.tagged_words(categories="news")]
common_words=[]
for word in word_tokenize(barack):
    if word in words:
        common_words.append(word)
# common_words
'Illinois' in words
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
tagged_barack = unigram_tagger.tag(tokenised_barack)
# print(tagged_barack)
unigram_tagger.evaluate([pos_list])
# pos_list
words.index("United")
words[7155],words[7156],words[7157],words[7158]
from nltk.corpus import wordnet as wn
wn.synset('dog.n.01').hypernyms()
wn.synset('cat.n.01').hypernyms()
wn.synset('animal.n.01').hypernyms()
dog = wn.synset('dog.n.01')
dog.lemmas()
dog.hyponyms()
#path similariy method shortest path in the taxonomy
cat = wn.synset('cat.n.01')
cat.path_similarity(dog)
#dog and cat synset similarity
#wu and palmer similarity method based on their Least Common Subsumer and the maximum depth in the taxonomy
cat.wup_similarity(dog)
animal = wn.synset('animal.n.01')
animal.path_similarity(wn.synset('bird.n.01'))
animal.wup_similarity(wn.synset('bird.n.01'))
#access to all synsets
# list(wn.all_synsets('n'))
#MORPHY
wn.morphy("working" , wn.VERB)
wn.morphy("denied" , wn.VERB)
wn.morphy("abaci")
#synonyms
synonyms = []
for syn in wn.synsets("good"):
    for word in syn.lemmas():
        if word.name() != "good":
            synonyms.append(word.name())
#print(synonyms)
#antonyms
antonyms = []
for syn in wn.synsets("good"):
    for word in syn.lemmas():
        if word.name() != "good" and word.antonyms() :
            antonyms.append( word.antonyms()[0].name())
# print(antonyms)
from nltk.sentiment import SentimentIntensityAnalyzer
sa = SentimentIntensityAnalyzer()
print(sa.polarity_scores("very bad"))
import pandas as pd 
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
sms_data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')
sms_data.head()
sms_data.columns
cols = sms_data.columns[:2]
data = sms_data[cols]
data.shape
data = data.rename(columns={"v1":"Value","v2":"Text"})
data.head()
data.Value.value_counts()
data['Value_num'] = data.Value.map({"spam":1,"ham":0})
# X = data[""] 
# data[data["Value"]=="spam"]
from string import punctuation
from re import *
import nltk
from nltk import word_tokenize

punctuation = list(punctuation)
# print(punctuation)
#no of punctuations
data["Punctuations"] = data["Text"].apply(lambda x: len(re.findall(r"[^\w+&&^\s]",x)))
# len(re.findall(r"[^\w+&&^\s]",'Go until jurong point, crazy.. Available only ...'))

#no of punctuations
data["Phonenumbers"] = data["Text"].apply(lambda x: len(re.findall(r"[0-9]{10}",x)))
# print(len(re.findall(r"[0-9]{10}","9999999999")))

#links
is_link = lambda x: 1 if re.search(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+",x)!=None else 0
data["Links"] = data["Text"].apply(is_link)
# print(re.search(r"(http|https)?(://)?(www.)?.*(.)[A-Za-z]+","http://www.a.cm"))
# print(is_link("http://hello.c"))

is_link = lambda x : list(map(str.isupper,x.split())).count(True) 
upper_case = lambda y,n : n+1 if y.isupper() else n
data["Uppercase"] = data["Text"].apply(is_link)
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return len(sorted(unusual))
data["unusualwords"] = data["Text"].apply(lambda x: unusual_words(word_tokenize(x)))
data[14:25]
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf= TfidfVectorizer(stop_words="english",strip_accents='ascii',max_features=300)
tf_idf_matrix = tf_idf.fit_transform(data["Text"])
data_extra_features = pd.concat([data,pd.DataFrame(tf_idf_matrix.toarray(),columns=tf_idf.get_feature_names())],axis=1)
X=data_extra_features
features = X.columns.drop(["Value","Text","Value_num"])
target = ["Value"]
X_train,X_test,y_train,y_test = train_test_split(X[features],X[target])
X_train.shape
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
pred = dt.predict(X_test)
print(accuracy_score(y_train, dt.predict(X_train)))
print(accuracy_score(y_test, pred))
print(dt.tree_.node_count)
my_submission=pd.DataFrame({'ID': test.Id, 'SalePrice': pred})
my_submission.to_csv('submission.csv',index=False)
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
pred_mnb = mnb.predict(X_test)

print(accuracy_score(y_test, pred_mnb))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred_lr = lr.predict(X_test)

print(accuracy_score(y_test, pred_lr))
X = data["Text"]
y =  data["Value_num"]
#vectorize the train and test matrices for label Text
cnt_vect = CountVectorizer()
X_cnt_vect = cnt_vect.fit_transform(X)

#bag of words
# print(cnt_vect.get_feature_names()[:20])
# print(cnt_vect.get_feature_names()[-20:])
print(X_cnt_vect[:5,:1000])
tfidf_vect = TfidfVectorizer(strip_accents='unicode', stop_words="english",)
X_tfidf_vect = tfidf_vect.fit_transform(X)
from nltk.stem import SnowballStemmer
import re
l = tfidf_vect.get_feature_names()
feature_names = tfidf_vect.get_feature_names()
stemmed_list= []
stemObj = SnowballStemmer("english")
for i in l:
    if re.search(r".*[A-Za-z].*",i):
        stemmed_list.append(stemObj.stem(i))
list(set(stemmed_list))[-20:]
# print(X_tfidf_vect[:5,:1000])
# print(len(feature_names))
#  X_tfidf_vect.toarray()
print(X_cnt_vect.shape)
print(X_tfidf_vect.shape)
print(y.shape)
#create train X
X_train,X_test,y_train,y_test = train_test_split(X_cnt_vect,y, test_size = 0.25,random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
pred_cnt_vect = dt.predict(X_test)
# print(pred_cnt_vect)
accuracy_score(y_test, pred_cnt_vect)
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
pred_mnb_tfidf = mnb.predict(X_test)
# print(pred_mnb_tfidf)
accuracy_score(y_test, pred_mnb_tfidf)
train_toks = pd.DataFrame(X_train.toarray(),columns=tfidf_vect.get_feature_names())
# feature_set = pd.DataFrame(X_train.toarray(),columns=tfidf_vect.get_feature_names())
# def maxent_classifier_method(algo):
#      mxclassifier = MaxentClassifier.train(train_toks,algorithm=algo, trace=0, max_iter=1000)
# maxent_classifier_method("GIS")
# train_toks
# dict(tfidf_vect.get_feature_names(),X_train.toarray())
feature_set = train_toks.to_dict()
print(feature_set)
X_train,X_test,y_train,y_test = train_test_split(X_tfidf_vect,y, test_size = 0.25,random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

pred_tfidf_vect = dt.predict(X_test)
# print(pred_cnt_vect)

accuracy_score(y_test, pred_tfidf_vect)
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
pred_mnb_tfidf = mnb.predict(X_test)
# print(pred_mnb_tfidf)
accuracy_score(y_test, pred_mnb_tfidf)
pred_mnb_tfidf = mnb.predict(X_test)
# print(pred_mnb_tfidf)
accuracy_score(y_test, pred_mnb_tfidf)
print(X_train.shape)
print(y_train.shape)
from nltk.classify import MaxentClassifier
def maxent_classifier_method(algo):
    mxclassifier = MaxentClassifier.train(data,algorithm=algo, trace=0, max_iter=1000)
maxent_classifier_method("GIS")
# X_test.shape
# for i in X_test:
#     print(i[0])
X_test[0:5,0:1000]
import nltk
from nltk.chat.util import Chat, reflections
from sys import version_info
from string import punctuation 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

pairs = [
    [
        r"How can I avail internet reservation facility through credit cards?",
        ['Recently internet reservation facility has started on Indian Railways. The web site http://www.irctc.co.in is operational, wherein you can get the railway reservation done through Credit Cards.For more on Reservation through credit cards click here  Internet Reservation',]
    ],
    [
        r'Why are PNR and reservation availability queries not available after certain timings at night?',
        ['The online PNR and seat availability queries are fetched from the computerized reservation applications. These online reservation applications are shut down daily around 2330 hrs to 0030 hrs IST. Due to the dynamic changes taking place in the PNR status updation and the availability positions, these two types of queries have to be fetched from the online reservation applications, hence the non- availability of them after certain timings. The sheer size of these databases does not allow them to be copied over network lines.Please note that the web site is functional 24 hrs. a day and other queries (trains between any two stations, fare queries, etc.) are functional throughout the day.',]
    ],
    [
    r'How can I avail the enquiries, through SMS on mobile phones?',
    ['Now all the enquiries offered on the web site www.indianrail.gov.in are available on your mobile phone through SMS facility. For more information on the mobile service providers and the key words to be used on the mobile, please click here, SMS help . Please note that we are giving the backend service only for the SMS queries. For more information and help on key words and SMS facility, kindly contact the mobile service provider according to the table.',]
    ],
    [
    r'Why do sometimes the fonts, colors schemes and java scripts behave differently in some browser or browsers?',
    ['This web site is best viewed with Microsoft Internet Explorer 6.0 and above. It might not give desired results with other browsers. All the pages, color schemes and scripts have been tested for IE 6.0 and above. ',]

    ],
    [
    r'Where can I get the latest arrival and departure timings of trains, when they get delayed?',
    ['The latest arrival and departure timings of delayed trains, alongwith diverted routes etc. will be made available shortly on this web site only.',]
    ],
    [
    r'Where can I lodge complaint against any type of grievances in the Trains, Platforms, officials for problems on this web site and give suggestions?',
    ['The complaint software is presently under development. We try our best to forward your grievances to the concerned department. However please note that this is not always possible. Please note that all your complaints and suggestions for the improvement of the web site http://www.indianrail.gov.in  can be put on the Feedback & suggestions page. Please note that, in case of any problems, give the query type (hyper link), the inputs which you gave, and the exact error message generated by this web site. All this will help us in solving the problems quickly. In the absence of such inputs, we cannot solve the problems.',]
    ],

]
def unique(list1):
    # intilize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return(unique_list)
lemmatiser = WordNetLemmatizer()
def preprocessing (sent) :
    rem_words = ['get', 'avail', 'who' , 'where', 'how' , 'what', 'why' , 'when', 'I', 'can']
    ##print(sent)
    # remove punctuation
    # convert to lower
    for p in list(punctuation):
        sent=sent.replace(p,'')
    sent=sent.lower().split()
    #remove stop words 
    stop_words = set(stopwords.words('english'))
    sent = [w for w in sent if not w in stop_words]
    sent = [w for w in sent if not w in rem_words]
    # lemmitise 
    #[item.upper() for item in mylis]
    sent = [lemmatiser.lemmatize(item, pos="v") for item  in sent ]
    return(unique(sent)) 


def tellme_bot():
    ##print("Hi how can I help you today?")
    while(1):
        #print('\n' *10 )
        response = input("Tell Me. [q to quit]>")
        if response == 'q':
            break
        i=0
        chosen = len(pairs) 
        matches = 0
        list_response=preprocessing(response) 
        #print(list_response)
        ##print(list_response)
        while ( i < len(pairs) ):
            #print("The idx is : " + str(i) )
            #print('----------------------------------------------------')
            loc_matches = 0 
            #x=pairs[i][0]
            x=pairs[i][0] + "  ".join(pairs[i][1])
            #y=x[0]
            #print(x)
            list_pair=preprocessing(x)
            #print(list_pair)
            for word in list_pair:
                if word in list_response:
                    #print (word+ ' is a  word in pairs')
                    loc_matches=loc_matches+1
            #print('loc matches :'+ str(i) + " " +str(loc_matches) ) 
            if ( loc_matches > matches ):
                chosen = i 
                matches = loc_matches
            i = i + 1 
        if ( chosen <len(pairs) ) :
            ans=pairs[chosen][1]
            print(ans[0] ) 
        else :
            print("Unable to answer this question" ) 
        break
tellme_bot()
#10 Ways Sugar Harms Your Health
# https://www.atkins.com/how-it-works/library/articles/10-ways-sugar-harms-your-health

docs1="Sugar causes blood glucose to spike and plummet. Unstable blood sugar often leads to mood swings, fatigue, headaches and cravings for more sugar. Cravings set the stage for a cycle of addiction in which every new hit of sugar makes you feel better temporarily but, a few hours later, results in more cravings and hunger. On the flip side, those who avoid sugar often report having little or no cravings for sugary things and feeling emotionally balanced and energized."
docs2="Sugar increases the risk of obesity, diabetes and heart disease. Large-scale studies have shown that the more high-glycemic foods (those that quickly affect blood sugar), including foods containing sugar, a person consumes, the higher his risk for becoming obese and for developing diabetes and heart disease1. Emerging research is also suggesting connections between high-glycemic diets and many different forms of cancer."
docs3="Sugar interferes with immune function. Research on human subjects is scant, but animal studies have shown that sugar suppresses immune response5. More research is needed to understand the exact mechanisms; however, we do know that bacteria and yeast feed on sugar and that, when these organisms get out of balance in the body, infections and illness are more likely."
docs4="A high-sugar diet often results in chromium deficiency. Its sort of a catch-22. If you consume a lot of sugar and other refined carbohydrates, you probably dont get enough of the trace mineral chromium, and one of chromiums main functions is to help regulate blood sugar. Scientists estimate that 90 percent of Americans dont get enough chromium. Chromium is found in a variety of animal foods, seafood and plant foods. Refining starches and other carbohydrates rob these foods of their chromium supplies."
docs5="Sugar accelerates aging. It even contributes to that telltale sign of aging: sagging skin. Some of the sugar you consume, after hitting your bloodstream, ends up attaching itself to proteins, in a process called glycation. These new molecular structures contribute to the loss of elasticity found in aging body tissues, from your skin to your organs and arteries7. The more sugar circulating in your blood, the faster this damage takes hold."
docs6="Sugar causes tooth decay. With all the other life-threatening effects of sugar, we sometimes forget the most basic damage it does. When it sits on your teeth, it creates decay more efficiently than any other food substance8. For a strong visual reminder, next time the Tooth Fairy visits, try the old tooth-in-a-glass-of-Coke experiment—the results will surely convince you that sugar isnt good for your pearly whites."
docs7="Sugar can cause gum disease, which can lead to heart disease. Increasing evidence shows that chronic infections, such as those that result from periodontal problems, play a role in the development of coronary artery disease9. The most popular theory is that the connection is related to widespread effects from the bodys inflammatory response to infection."
docs7="Sugar affects behavior and cognition in children. Though it has been confirmed by millions of parents, most researchers have not been able to show the effect of sugar on childrens behavior. A possible problem with the research is that most of it compared the effects of a sugar-sweetened drink to one containing an artificial sweetener10. It may be that kids react to both real sugar and sugar substitutes, therefore showing no differences in behavior. What about kids ability to learn? Between 1979 and 1983, 803 New York City public schools reduced the amount of sucrose (table sugar) and eliminated artificial colors, flavors and two preservatives from school lunches and breakfasts. The diet policy changes were followed by a 15.7 percent increase in a national academic ranking (previously, the greatest improvement ever seen had been 1.7 percent)."
docs8="Sugar increases stress. When were under stress, our stress hormone levels rise; these chemicals are the bodys fight-or-flight emergency crew, sent out to prepare the body for an attack or an escape. These chemicals are also called into action when blood sugar is low. For example, after a blood-sugar spike (say, from eating a piece of birthday cake), theres a compensatory dive, which causes the body to release stress hormones such as adrenaline, epinephrine and cortisol. One of the main things these hormones do is raise blood sugar, providing the body with a quick energy boost. The problem is, these helpful hormones can make us feel anxious, irritable and shaky."
docs9="Sugar takes the place of important nutrients. According to USDA data, people who consume the most sugar have the lowest intakes of essential nutrients––especially vitamin A, vitamin C, folate, vitamin B-12, calcium, phosphorous, magnesium and iron. Ironically, those who consume the most sugar are children and teenagers, the individuals who need these nutrients most12."
docs10="Slashing Sugar. Now that you know the negative impacts refined sugar can have on your body and mind, youll want to be more careful about the foods you choose. And the first step is getting educated about where sugar lurks—believe it or not, a food neednt even taste all that sweet for it to be loaded with sugar. When it comes to convenience and packaged foods, let the ingredients label be your guide, and be aware that just because something boasts that it is low in carbs or a diet food, doesnt mean its free of sugar. Atkins products never contain added sugar."


# compile documents
doc_complete=[docs1,docs2,docs3, docs4,docs5,docs6,docs7,docs8,docs9,docs10, ]
#print(doc_complete)

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete] 
#print(doc_clean)
# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
#print(doc_term_matrix)

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=300)

#Result
#print(ldamodel.print_topics(num_topics=5, num_words=5))
topics = ldamodel.print_topics(num_topics=5, num_words=5)
for  i in topics :
    print (i) 
    
