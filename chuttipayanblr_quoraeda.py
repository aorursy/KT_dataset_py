import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
traindata=pd.read_csv('../input/train.csv')
traindata.shape
traindata.info()
traindata.target.value_counts().plot(kind='bar')
import nltk
from nltk.tokenize import word_tokenize
traindata['wordlen']=traindata['question_text'].apply(lambda x: len(word_tokenize(x)))
traindata.head()
sns.boxplot(x='target',y='wordlen',data=traindata)
print("Max,Mean and Min of word count for Sincere questions")
print(traindata[traindata.target==0]['wordlen'].max())
print(traindata[traindata.target==0]['wordlen'].mean())
print(traindata[traindata.target==0]['wordlen'].min())
print("Query with maximum word count", traindata[(traindata.target==0) & (traindata.wordlen==traindata[traindata.target==0]['wordlen'].max())]['question_text'])
                                                 
print("Query with Minimum word count", traindata[(traindata.target==0) & (traindata.wordlen==traindata[traindata.target==0]['wordlen'].min())]['question_text'])
                                                 
                                                 
print("Max,Mean and Min of word count for Sincere questions")
print(traindata[traindata.target==1]['wordlen'].max())
print(traindata[traindata.target==1]['wordlen'].mean())
print(traindata[traindata.target==1]['wordlen'].min())
print("Query with maximum word count", traindata[(traindata.target==1) & (traindata.wordlen==traindata[traindata.target==1]['wordlen'].max())]['question_text'])
                                                 
print("Query with Minimum word count", traindata[(traindata.target==1) & (traindata.wordlen==traindata[traindata.target==1]['wordlen'].min())]['question_text'])
                                                 
                                                 
traindata['sentencelen']=traindata['question_text'].apply(lambda x: len(x))
print("Max,Mean and Min of word count for Sincere questions")
print(traindata[traindata.target==0]['sentencelen'].max())
print(traindata[traindata.target==0]['sentencelen'].mean())
print(traindata[traindata.target==0]['sentencelen'].min())
print("Max,Mean and Min of word count for Sincere questions")
print(traindata[traindata.target==1]['sentencelen'].max())
print(traindata[traindata.target==1]['sentencelen'].mean())
print(traindata[traindata.target==1]['sentencelen'].min())
traindata.head()
sns.boxplot(x='target',y='sentencelen',data=traindata)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

traindata['sentiment']=traindata['question_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
sns.boxplot(x='target',y='sentiment',data=traindata)
sns.distplot(traindata[traindata.target==0]['sentiment'])
sns.distplot(traindata[traindata.target==1]['sentiment'])
from wordcloud import WordCloud
from nltk.corpus import stopwords
import re
from tqdm import tqdm
def preprocess_narrative( questions ):
    final=""
    for text in tqdm(questions):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = text.lower().split()  
        stops = set(stopwords.words("english")) 
        for w in words:
            if w not in stops:
                final=final+" "+w
    #print(final)
    return final

x=preprocess_narrative(traindata[traindata.target==1]['question_text'])
wc = WordCloud(background_color="white", max_words=1000,width=1000, height=500)# mask=alice_mask)
wc.generate(x)
fig = plt.figure(figsize = (10, 10))
plt.imshow(wc)
# sampling the data set for 0 as the data set is huge 
tempdata=traindata[traindata.target==0]
y=preprocess_narrative(tempdata.sample(frac=0.1)['question_text'])
wc.generate(y)
fig = plt.figure(figsize = (10, 10))
plt.imshow(wc)
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import regex as re
from tqdm import tqdm
# setup tokenizer
tokenizer = WordPunctTokenizer()

stops = set(stopwords.words("english"))
def text_to_wordlist(text, lower=False):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
    # Tokenize
    text = tokenizer.tokenize(text)
    
    # optional: lower case
    if lower:
        text = [t.lower() for t in text]
    
    
    text = [t if t not in stops else None for t in text]
    
    
    
    # Return a list of words
    vocab.update(text)
    #return text

def process_comments(list_sentences, lower=False):
    comments = []
    for text in tqdm(list_sentences):
        text_to_wordlist(text, lower=lower)
        
vocab=Counter()
process_comments(traindata[traindata.target==0]['question_text'],True)
vocab.pop(None)
since_most_common=vocab.most_common(20)
vocab=Counter()
process_comments(traindata[traindata.target==1]['question_text'],True)
vocab.pop(None)
insincere_most_common=vocab.most_common(20)
sincere_mc=pd.DataFrame(since_most_common)
insincere_mc=pd.DataFrame(insincere_most_common)
sincere_mc.columns=['word','count']
insincere_mc.columns=['word','count']

sincere_mc.plot(x='word',kind='bar')
insincere_mc.plot(x='word',kind='bar')

vocab=Counter()
process_comments(traindata[(traindata.target==0) & (traindata.sentiment <0)]['question_text'],True)
vocab.pop(None)
negative_sincere_most_common=vocab.most_common(20)
negative_sincere_most_common=pd.DataFrame(negative_sincere_most_common)
negative_sincere_most_common.columns=['word','count']
negative_sincere_most_common.plot(x='word',kind='bar')

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import regex as re
from tqdm import tqdm
# setup tokenizer
tokenizer = WordPunctTokenizer()
vocab=Counter()
org=Counter()
stops = set(stopwords.words("english"))
labels=[]
def process_ner(list_sentences):
    for text in tqdm(list_sentences):
        
        doc = nlp(text)
        for x in doc.ents:
            if(x.label_=='PERSON'):
                vocab.update([x.text.lower()])
            if(x.label_=='ORG'):
                org.update([x.text.lower()])
process_ner(traindata[traindata.target==1].sample(frac=0.5)['question_text'])
plt.figure(figsize=(20,20))
person_most_common=pd.DataFrame(vocab.most_common(50))
person_most_common.columns=['Name','count']
personplot=sns.barplot(y="count",x="Name",data=person_most_common)
loc, labels = plt.xticks(rotation='vertical')

plt.figure(figsize=(20,20))
org_most_common=pd.DataFrame(org.most_common(50))
org_most_common.columns=['Name','count']
orgplot=sns.barplot(y="count",x="Name",data=org_most_common)
loc, labels = plt.xticks(rotation='vertical')

