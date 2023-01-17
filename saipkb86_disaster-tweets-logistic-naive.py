import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import string

import nltk

#nltk.download('stopwords')

#nltk.download('punkt')

from nltk.corpus import stopwords

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV

from sklearn.svm import SVC

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import  Counter

from nltk.tokenize import word_tokenize

import re

import warnings 



stop=set(stopwords.words('english'))

warnings.filterwarnings("ignore")

%matplotlib inline
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
print('Train Data size :{}'.format(train_df.shape))

print('Test Data size :{}'.format(test_df.shape))
train_df.head()
train_df.isna().sum()
test_df.isna().sum()
train_df.groupby('location')['id'].count()
sns.set_style('darkgrid')

plt.figure(figsize=(6,6))

x=train_df['target'].value_counts()

sns.barplot(x.index,x)
fig,axes = plt.subplots(1,2,figsize=(12,6))

char_len = train_df[train_df['target']==1]['text'].str.len()

sns.distplot(char_len,ax=axes[0],kde=False)



char_len = train_df[train_df['target']==0]['text'].str.len()

sns.distplot(char_len,ax=axes[1],kde=False)
fig,axes= plt.subplots(2,1,figsize=(18,12))



temp = pd.DataFrame(train_df[train_df['target']==0].groupby('keyword')['id'].count())

temp.sort_values('id',ascending=False,inplace=True)

sns.barplot(temp.index[:10],temp['id'][:10],ax=axes[0]).set_title('Normal Tweets')





temp = pd.DataFrame(train_df[train_df['target']==1].groupby('keyword')['id'].count())

temp.sort_values('id',ascending=False,inplace=True)

sns.barplot(temp.index[:10],temp['id'][:10],ax=axes[1]).set_title('Disaster Tweets')
# Referenec : https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/



def hashtag_extract(x):

    hashtags = []

    # Loop over the words in the tweet

    for i in x:

        ht = re.findall(r"#(\w+)", i)

        hashtags.append(ht)



    return hashtags



HT_regular = hashtag_extract(train_df['text'][train_df['target'] == 0])



# extracting hashtags from racist/sexist tweets

HT_disaster = hashtag_extract(train_df['text'][train_df['target'] == 1])



# unnesting list

HT_regular = sum(HT_regular,[])

HT_disaster = sum(HT_disaster,[])
fig,axes = plt.subplots(2,1,figsize=(18,10))



a = nltk.FreqDist(HT_regular)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})

# selecting top 10 most frequent hashtags     

d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(16,5))

sns.barplot(data=d, x= "Hashtag", y = "Count",ax=axes[0]).set_title('Normal Tweets')





a = nltk.FreqDist(HT_disaster)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})

# selecting top 10 most frequent hashtags     

d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(16,5))

sns.barplot(data=d, x= "Hashtag", y = "Count",ax=axes[1]).set_title('Disaster Tweets')



Merge_df = train_df.append(test_df,ignore_index=True)
def remove_pattern(input_txt, pattern):

    reg_obj = re.compile(pattern)

    input_txt = reg_obj.sub(r'', input_txt)

        

    return input_txt   
Merge_df['text'] = Merge_df['text'].apply(lambda x: remove_pattern(x,"@[\w]*"))
# Reference : https://www.kaggle.com/shahules/tweets-complete-eda-and-basic-modeling



Merge_df['text'] = Merge_df['text'].apply(lambda x: remove_pattern(x,'https?://\S+|www\.\S+'))

Merge_df['text'] = Merge_df['text'].apply(lambda x: remove_pattern(x,'<.*?>'))

    
Merge_df['text'] = Merge_df['text'].apply(lambda x: remove_pattern(x,"[^a-zA-Z# ]"))
def remove_stop_words(text):

    

    word_tokens = word_tokenize(text) 

  

    filtered_sentence = [w for w in word_tokens if not w in stop] 

    

    filtered_tweet = ' '.join(filtered_sentence)

    

    return filtered_tweet
Merge_df['text'] = Merge_df['text'].apply(lambda x: remove_stop_words(x))
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()



def tokenize_stem(text):

    

    token_words = word_tokenize(text)

    stem_words =[]

    for i in token_words:

        word = lemmatizer.lemmatize(i)

        stem_words.append(word)

        

    final_tweet = ' '.join(stem_words)

    

    return final_tweet
Merge_df['text'] = Merge_df['text'].apply(lambda x: tokenize_stem(x))
all_words = ' '.join([text for text in Merge_df['text'][Merge_df['target']==0]])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(16, 10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
all_words = ' '.join([text for text in Merge_df['text'][Merge_df['target']==1]])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(16, 10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=300, stop_words='english')

# TF-IDF feature matrix

tfidf = tfidf_vectorizer.fit_transform(Merge_df['text'])
tfidf.shape
Final_train = tfidf[:7613]

Final_test = tfidf[7613:]
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



xtrain = Final_train[:5331]

xvalid = Final_train[5331:]

ytrain = Merge_df[:5331]['target']

yvalid = Merge_df[5331:7613]['target']



parameter = {'solver':['liblinear','lbfgs'],

            'max_iter':[200,400]}



Logis_clf = LogisticRegression()



lreg = GridSearchCV(Logis_clf, param_grid = parameter, cv = 3, verbose=True, n_jobs=-1)

lreg.fit(xtrain, ytrain) # training the model



prediction = lreg.predict_proba(xvalid) # predicting on the validation set
from sklearn.metrics import roc_auc_score,roc_curve,f1_score, confusion_matrix





# keep probabilities for the positive outcome only

lr_probs = prediction[:, 1]

lr_auc = roc_auc_score(yvalid, lr_probs)





print('ROC AUC=%.3f' % (lr_auc))



lr_fpr, lr_tpr, _ = roc_curve(yvalid, lr_probs)



# plot the roc curve for the model

plt.figure(figsize=(10,8))

plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()
prediction_int = prediction[:,1] >= 0.35

prediction_int = prediction_int.astype(np.int)

f1score = f1_score(yvalid,prediction_int)

print('F1 Score : %.3f' %(f1score))

conf = confusion_matrix(yvalid,prediction_int)

print(conf)
prediction = lreg.predict_proba(Final_test) # predicting on the test set



prediction_int = prediction[:,1] >= 0.35

prediction_int = prediction_int.astype(np.int)
NB_Clf = MultinomialNB()

NB_Clf.fit(xtrain, ytrain)



pred_naive = NB_Clf.predict(xvalid)

conf = confusion_matrix(yvalid, pred_naive)

print(conf)



f1score = f1_score(yvalid,pred_naive)

print('F1 Score : %.3f' %(f1score))



pred_naive_test = NB_Clf.predict(Final_test)

pred_naive_test = pred_naive_test.astype(int)



param_grid = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']}

SVM_Model = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, n_jobs=-1).fit(xtrain, ytrain)
pred_SVM_valid = SVM_Model.predict(xvalid)

pred_SVM_valid = pred_SVM_valid.astype(int)



conf = confusion_matrix(yvalid, pred_SVM_valid)

print(conf)



f1score = f1_score(yvalid,pred_SVM_valid)

print('F1 Score : %.3f' %(f1score))
test_df['target'] = prediction_int



Final_submission= test_df[['id','target']]

Final_submission.to_csv('submission.csv',index=False)