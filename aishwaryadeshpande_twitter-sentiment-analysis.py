# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re    # for regular expressions 

import nltk  # for text manipulation 

import string 

import warnings

import seaborn as sns 

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import the modules we'll need

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index = False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
train = pd.read_csv('/kaggle/input/twitter-tweets/train_2kmZucJ (1).csv')

test = pd.read_csv('/kaggle/input/twitter-tweets/test_oJQbWVk.csv')
train[train['label'] == 0].head()
train[train['label'] == 1].head()
train['label'].value_counts()
train_length = train['tweet'].str.len()

test_length = test['tweet'].str.len()



plt.hist(train_length,bins = 20,label = 'train_length')

plt.hist(test_length,bins = 20,label = 'test_length')

plt.legend()

plt.show()

combi = train.append(test, ignore_index=True) 

combi.shape
def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

    return input_txt    
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")

combi.head()
for i in range(len(combi['tidy_tweet'])) :

    combi['tidy_tweet'][i] = re.sub(r'http\S+',' ',combi['tidy_tweet'][i])
for i in combi['tidy_tweet']:

    print(i)
combi['tidy_tweet']  = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ") 

combi.head(10)
combi['tidy_tweet']  = combi['tidy_tweet'].apply(lambda x : ' '.join([w for w in x.split() if len(w)>3]))

combi.head(10)
from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

  

stop = stopwords.words('english')



tokenized_tweet = combi['tidy_tweet'].apply(lambda x : x.split())

tokenized_tweet.apply(lambda x: [item for item in x if item not in stop])

for i in range(len(tokenized_tweet)):

    tokenized_tweet[i] = ' '.join(tokenized_tweet[i]) 

combi['tidy_tweet'] = tokenized_tweet
tokenized_tweet = combi['tidy_tweet'].apply(lambda x : x.split()) #tokenization 

tokenized_tweet.head()
#normalizing the tokenized tweets

from nltk.stem.porter import * 

from nltk.stem import WordNetLemmatizer 



stemmer = PorterStemmer() 

lemmatizer = WordNetLemmatizer()

#tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x]) # stemming
for i in range(len(tokenized_tweet)):

    tokenized_tweet[i] = ' '.join(tokenized_tweet[i]) 

combi['tidy_tweet'] = tokenized_tweet
from wordcloud import WordCloud



all_words = ' '.join([text for text in combi['tidy_tweet']])  

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 



plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
nonracist_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])

nonracist_word_cloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(nonracist_words)

                            

plt.figure(figsize=(10, 7)) 

plt.imshow(nonracist_word_cloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
all_racist_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])

racist_word_cloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_racist_words)

                            

plt.figure(figsize=(10, 7)) 

plt.imshow(racist_word_cloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
#function to collect hastags

def hashtagExtract(x):

    hashtags = []

    for i in x:

        ht = re.findall(r"#(\w+)", i)        

        hashtags.append(ht) 

    return hashtags
#extracting hashtags from non racist comments

HT_nonracist = hashtagExtract(combi['tidy_tweet'][combi['label'] == 0]) 

#extracting hashtags from racist  comments

HT_negative = hashtagExtract(combi['tidy_tweet'][combi['label'] == 1]) 

#unnesting

HT_nonracist = sum(HT_nonracist,[])

HT_negative = sum(HT_negative,[])
a = nltk.FreqDist(HT_nonracist) 

d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())}) 

# selecting top 20 most frequent hashtags     

d = d.nlargest(columns="Count", n = 20) 



plt.figure(figsize=(16,5)) 

ax = sns.barplot(data=d, x= "Hashtag", y = "Count") 

ax.set(ylabel = 'Count') 

plt.show()
b = nltk.FreqDist(HT_negative)

e = pd.DataFrame({'Hashtag':list(b.keys()),'Count': list(b.values())})



e = e.nlargest(columns="Count", n = 20) 



plt.figure(figsize=(16,5)) 

ax = sns.barplot(data=e, x= "Hashtag", y = "Count") 

ax.set(ylabel = 'Count') 

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 

import gensim

from nltk.tokenize import TreebankWordTokenizer



bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2,  max_features=1500,stop_words='english') 



tokenizer = TreebankWordTokenizer()

bow_vectorizer.set_params(tokenizer=tokenizer.tokenize)



# include 1-grams and 2-grams

bow_vectorizer.set_params(ngram_range=(1, 3))



# ignore terms that appear in more than 50% of the documents

bow_vectorizer.set_params(max_df=0.5)



# only keep terms that appear in at least 2 documents

bow_vectorizer.set_params(min_df=2)

bow = bow_vectorizer.fit_transform(combi['tidy_tweet']) 

bow.shape
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

tokenizer = TreebankWordTokenizer()

tfidf_vectorizer.set_params(tokenizer=tokenizer.tokenize)



# include 1-grams and 2-grams

tfidf_vectorizer.set_params(ngram_range=(1, 2))



# ignore terms that appear in more than 50% of the documents

tfidf_vectorizer.set_params(max_df=0.5)
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet']) 

tfidf.shape
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing 

model_w2v = gensim.models.Word2Vec(tokenized_tweet,size=200,window=5,min_count=2,sg = 1,hs = 0,negative = 10,workers= 2,seed = 34)



model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)
model_w2v.wv.most_similar(positive="food")
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
wordvec_df.shape
from tqdm import tqdm 

tqdm.pandas(desc="progress-bar") 

from gensim.models.doc2vec import LabeledSentence
def add_label(twt):

    output = []

    for i, s in zip(twt.index, twt):

        output.append(LabeledSentence(s, ["tweet_" + str(i)]))

    return output

labeled_tweets = add_label(tokenized_tweet) # label all the tweets
labeled_tweets[:6]
model_d2v = gensim.models.Doc2Vec(dm=1, dm_mean=1, size=200,window=5,negative=7, min_count=5,workers=3,alpha=0.1,seed = 23) 

model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])

model_d2v.train(labeled_tweets, total_examples= len(combi['tidy_tweet']), epochs=15)
docvec_arrays = np.zeros((len(tokenized_tweet), 200)) 

for i in range(len(combi)):

    docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1,200))    



docvec_df = pd.DataFrame(docvec_arrays) 

docvec_df.shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split 

from sklearn.metrics import f1_score

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn import model_selection

from mlxtend.classifier import EnsembleVoteClassifier


def ModelDevelopment(feat,test_feat,model):

    x_train,x_valid,y_train,y_valid = train_test_split(feat, train['label'],random_state=42,test_size=0.3)

    # training the model 

    model.fit(x_train, y_train) 



    prediction = model.predict_proba(x_valid) # predicting on the validation set 

    prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0 

    prediction_int = prediction_int.astype(np.int) 

    cross_val = model_selection.cross_val_score(model, x_train, y_train,cv=5, scoring='accuracy')

    print("Cross_Val"+str(cross_val))

    print(f1_score(y_valid, prediction_int))



    #Test Prediction 

    

    test_pred = model.predict_proba(test_feat) 

    test_pred_int = test_pred[:,1] >= 0.3 

    test_pred_int = test_pred_int.astype(np.int) 

    test['label'] = test_pred_int 

    submission = test[['id','label']]     

    

    return  submission
# Extracting train and test BoW features 

train_bow = bow[:7920,]

test_bow = bow[7920:9873,]
lreg = LogisticRegression() 

svc = svm.SVC(kernel='linear', C=1, probability=True)

rf = RandomForestClassifier(n_estimators=400, random_state=11)

xgb_model = XGBClassifier(max_depth=6, n_estimators=1000)

models = [lreg,svc,rf,xgb_model]

labels = ["logistic","svc","RandomForest","XGB"]
i = 0

for mod in models:

    print(labels[i])

    ModelDevelopment(train_bow,test_bow,mod)

    i = i+1
train_tf = tfidf[:7920,]

test_tf = tfidf[7920:9873,]

i = 0

for mod in models:

    print(labels[i])

    ModelDevelopment(train_tf,test_tf,mod)

    i = i+1
train_wod = wordvec_df.iloc[:7920,]

test_wod = wordvec_df.iloc[7920:9873,]

i = 0

for mod in models:

    print(labels[i])

    ModelDevelopment(train_wod,test_wod,mod)

    i = i+1
#best model till now with 0.9000 leadboard f1 score

sub = ModelDevelopment(train_wod,test_wod,xgb_model)

create_download_link(sub,"Download csv link",'xgb_model_w2v.csv')
train_doc = docvec_df.iloc[:7920,]

test_doc = docvec_df.iloc[7920:9873,]

i = 0

for mod in models:

    print(labels[i])

    ModelDevelopment(train_doc,test_doc,lreg)

    i = i+1
xgb_model
n_estimators = [500,1000]

for dep in n_estimators:

    xgb= XGBClassifier(max_depth=6, n_estimators=dep)

    ModelDevelopment(train_wod,test_wod,xgb)
grid = GridSearchCV(estimator=xgb_model, param_grid=params, cv=5)
eclf = EnsembleVoteClassifier(clfs=[lreg,svc,rf,xgb_model], weights=[1,1,1,1])



x_train,x_valid,y_train,y_valid = train_test_split(train_wod, train['label'],random_state=42,test_size=0.3)

labels = ['Logistic Regression','Support Vector Machine','Random Forest', 'XGB Model', 'Ensemble']

for clf, label in zip([lreg,svc,rf,xgb_model, eclf], labels):



    scores = model_selection.cross_val_score(clf, x_train, y_train,cv=5, scoring='accuracy')                                                                             

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
eclf = EnsembleVoteClassifier(clfs=[lreg,svc,rf,xgb_model], weights=[0.4,0.1,0.1,0.4])

sub = ModelDevelopment(train_wod,test_wod,eclf)

create_download_link(sub,"Download csv link",'ensemble.csv')
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB 

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import EnsembleVoteClassifier



clf1 = LogisticRegression(random_state=1)

clf2 = RandomForestClassifier(random_state=1)

clf3 = GaussianNB()

eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft')



params = {'logisticregression__C': [1.0, 100.0],'randomforestclassifier__n_estimators': [300, 400],}



grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)

grid.fit(x_train,y_train)



cv_keys = ('mean_test_score', 'std_test_score', 'params')



for r, _ in enumerate(grid.cv_results_['mean_test_score']):

    print("%0.3f +/- %0.2f %r"% (grid.cv_results_[cv_keys[0]][r], grid.cv_results_[cv_keys[1]][r] / 2.0,grid.cv_results_[cv_keys[2]][r]))