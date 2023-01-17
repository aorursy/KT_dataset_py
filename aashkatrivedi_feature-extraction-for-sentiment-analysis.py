#Loading Libraries

#text analysis
import re 
import nltk 
import string

#data processing
import pandas as pd
import numpy as np

#visualisation
import matplotlib.pyplot as plt 
import seaborn as sns



%matplotlib inline
pd.set_option("display.max_colwidth", 300)
#This helps us see the dataframes in a more visually pleasing manner
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head()
#Typical non-negative tweets

train[train['label'] == 0].head(10)
#Typical negative tweets
train[train['label'] == 1].head(10)
#Dataset shape
train.shape, test.shape
#Dataset Distribution

train["label"].value_counts()
#Visualising
%matplotlib inline

labels=['Negative', 'Positive']
colors = ['mistyrose','lightcyan']
sizes=[train['label'].value_counts()[1],
     train['label'].value_counts()[0]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, colors=colors,autopct='%1.0f%%')
ax1.axis('equal')
plt.show()
#Checking the distribution of length of tweets in training and test set

length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()

plt.hist(length_train, bins=20, label="train_tweets", color='mistyrose')
plt.hist(length_test, bins=20, label="test_tweets", color='darksalmon')
plt.legend()
plt.show()
#combine test and train set
combine= train.append(test,ignore_index=True, sort=True)
#function that removes a specific user-defined pattern from text, which we can use later
def remove_pattern(input_txt, pattern):
    r=re.findall(pattern,input_txt)
    for i in r:
        input_txt=re.sub(i,'',input_txt)
    return input_txt

#create a new column "tidy_tweet" which has our processed, tidy treat without redundancies
#removes '@user'; regular expression "@[\w]*" returns anything starting with @

combine['tidy_tweet'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")
#remove special characters, except #
combine['tidy_tweet'] = combine['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
#remove short words
combine['tidy_tweet']=combine['tidy_tweet'].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]))
combine.head()
#tokenising
tokenised_tweet= combine['tidy_tweet'].apply(lambda x: x.split())
tokenised_tweet.head()
#stemming
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenised_tweet = tokenised_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenised_tweet.head()
#put all tokens back as the tidy_tweet
for i in range(len(tokenised_tweet)):
    tokenised_tweet[i] = ' '.join(tokenised_tweet[i])
combine['tidy_tweet']=tokenised_tweet


combine.head()
#wordcloud generation for all tweets

all_words=' '.join([text for text in combine['tidy_tweet']])
from wordcloud import WordCloud
word_cloud=WordCloud(width=800, height=500, random_state=21,max_font_size=110,colormap='tab20').generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()
#wordcloud for positive (non-racist) tweets
non_negative_words=' '.join([text for text in combine['tidy_tweet'][combine['label']==0]])
from wordcloud import WordCloud
word_cloud=WordCloud(width=800, height=500, random_state=21,max_font_size=110,colormap='tab20').generate(non_negative_words)

plt.figure(figsize=(10, 7))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()
#wordcloud for negative (racist) tweets
negative_words=' '.join([text for text in combine['tidy_tweet'][combine['label']==1]])
from wordcloud import WordCloud
word_cloud=WordCloud(width=800, height=500, random_state=21,max_font_size=110,colormap='tab20').generate(negative_words)

plt.figure(figsize=(10, 7))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()
# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags
#hashtag list for non negative tweets
HT_non_negative = hashtag_extract(combine['tidy_tweet'][combine['label'] == 0])

#hashtag list for negative tweets
HT_negative = hashtag_extract(combine['tidy_tweet'][combine['label'] == 1])

#unnest list
HT_non_negative = sum(HT_non_negative,[])
HT_negative = sum(HT_negative,[])
#most used hashtag for non negative tweets

a = nltk.FreqDist(HT_non_negative)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(10,8))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count",palette="cubehelix")
ax.set(ylabel = 'Count')
plt.show()
#most negative hashtags
b = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag': list(b.keys()),
                  'Count': list(b.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(10,8))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count",palette="cubehelix")
ax.set(ylabel = 'Count')
plt.show()
#bag of word features
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow=bow_vectorizer.fit_transform(combine['tidy_tweet'])

bow.shape
#TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf=tfidf_vectorizer.fit_transform(combine['tidy_tweet'])

tfidf.shape
import gensim
tokenized_tweet = combine['tidy_tweet'].apply(lambda x: x.split()) # tokenizing

model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            size=200, # desired no. of features/independent variables 
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)

model_w2v.train(tokenized_tweet, total_examples= len(combine['tidy_tweet']), epochs=20)
#Find most similar words in the corpus for a given word

model_w2v.wv.most_similar(positive="trump")
def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
                         
            continue
    if count != 0:
        vec /= count
    return vec
wordvec_arrays = np.zeros((len(tokenized_tweet), 200))

for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)
    
wordvec_df = pd.DataFrame(wordvec_arrays)

wordvec_df.shape
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#BOG feature model

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

xtrain_bow, xval_bow,ytrain,yval=train_test_split(train_bow,train['label'],random_state=42,test_size=0.3)

lr=LogisticRegression()
lr.fit(xtrain_bow,ytrain)

prediction=lr.predict_proba(xval_bow)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yval,prediction_int)
test_pred = lr.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int

submission = test[['id','label']]

#submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file
#TF-IDF
train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yval.index]

lr.fit(xtrain_tfidf, ytrain)

prediction = lr.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yval, prediction_int)
train_w2v = wordvec_df.iloc[:31962,:]
test_w2v = wordvec_df.iloc[31962:,:]

xtrain_w2v = train_w2v.iloc[ytrain.index,:]
xvalid_w2v = train_w2v.iloc[yval.index,:]
lr.fit(xtrain_w2v, ytrain)

prediction = lr.predict_proba(xvalid_w2v)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
f1_score(yval, prediction_int)