#This is my first participation in kaggle competitions
#I wish evaluation and support
#thank forll all 
#this solution Divided into two main parts first clean the data and some visualisation
#then using gaussian naive bayes classifier  
#with evaliting the train data with confusion_matrix and root mean square error 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
from numpy import nan
from bs4 import BeautifulSoup    
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
train_data = pd.read_csv('../input/train.csv')


train_data.head()
#cleaning the data
train_data.info()
comment_text_train = []
for i in range(0,159571):
    review = re.sub('[^a-zA-Z]', ' ', train_data['comment_text'][i])
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    review = ' '.join(review)
    comment_text_train.append(review)
train_data['new_comment_text']=comment_text_train
train_data.drop(['comment_text'],axis=1,inplace=True)
train_data.head()
train_data=train_data[['id','new_comment_text','toxic','severe_toxic','obscene','threat','insult','identity_hate']]
#here i divide the data for visualisation
train_data.head()
toxic=train_data[['new_comment_text','toxic']]
toxic1=toxic[toxic['toxic']==(1)]
severe_toxic=train_data[['new_comment_text','severe_toxic']]
severe_toxic1=severe_toxic[severe_toxic['severe_toxic']==(1)]
obscene=train_data[['new_comment_text','obscene']]
obscene1=obscene[obscene['obscene']==(1)]
threat=train_data[['new_comment_text','threat']]
threat1=threat[threat['threat']==(1)]
insult=train_data[['new_comment_text','insult']]
insult1=insult[insult['insult']==(1)]
identity_hate=train_data[['new_comment_text','identity_hate']]
identity_hate1=toxic[identity_hate['identity_hate']==(1)]
words = ' '.join(toxic1['new_comment_text'])
split_word = " ".join([word for word in words.split()])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
words1 = ' '.join(severe_toxic1['new_comment_text'])
split_word1 = " ".join([word for word in words1.split()])
wordcloud1 = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word1)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()
words2 = ' '.join(obscene1['new_comment_text'])
split_word2 = " ".join([word for word in words2.split()])
wordcloud2 = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word2)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud2)
plt.axis('off')
plt.show()
words3 = ' '.join(threat1['new_comment_text'])
split_word3 = " ".join([word for word in words3.split()])
wordcloud3 = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word3)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud3)
plt.axis('off')
plt.show()
words4 = ' '.join(insult1['new_comment_text'])
split_word4 = " ".join([word for word in words4.split()])
wordcloud4 = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word4)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud4)
plt.axis('off')
plt.show()
words5 = ' '.join(identity_hate1['new_comment_text'])
split_word5 = " ".join([word for word in words5.split()])
wordcloud5 = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word5)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud5)
plt.axis('off')
plt.show()
#prepare data for count most word in this data set 
toxic_2=toxic1['new_comment_text']
severe_toxic2=severe_toxic1['new_comment_text']
obscene2=obscene1['new_comment_text']
threat2=threat1['new_comment_text']
insult2=insult1['new_comment_text']
identity_hate2=identity_hate1['new_comment_text']
vectorizer1 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 
vectorizer2 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 
vectorizer3 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 
vectorizer4 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 
vectorizer5 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 
vectorizer6 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 
toxic_feature = vectorizer1.fit_transform(toxic_2)
toxic_feature=toxic_feature.toarray()
toxic_feature.shape

toxic_feature_vectorize= vectorizer1.get_feature_names()
toxic_feature_vectorize
toxic_dist = np.sum(toxic_feature, axis=0)
for tag, count in zip(toxic_feature_vectorize, toxic_dist):
    print (tag,count)
toxic_new= pd.DataFrame(toxic_dist)
toxic_new.columns=['word_count']
toxic_new['word'] = pd.Series(toxic_feature_vectorize, index=toxic_new.index)
toxic_new_1=toxic_new[['word','word_count']]
toxic_new_top_15=toxic_new_1.sort_values(['word_count'],ascending=[0])
toxic_new_top_15.head(15)
severe_toxic_feature = vectorizer2.fit_transform(severe_toxic2)
severe_toxic_feature=severe_toxic_feature.toarray()
severe_toxic_feature.shape
severe_feature_vectorize= vectorizer2.get_feature_names()
severe_feature_vectorize
severe_dist = np.sum(severe_toxic_feature, axis=0)
for tag, count in zip(severe_feature_vectorize, severe_dist):
    print (tag,count)
severe_new= pd.DataFrame(severe_dist)
severe_new.columns=['word_count']
severe_new['word'] = pd.Series(severe_feature_vectorize, index=severe_new.index)
severe_new_1=severe_new[['word','word_count']]
severe_new_top_15=severe_new_1.sort_values(['word_count'],ascending=[0])
severe_new_top_15.head(15)
obscene_feature = vectorizer3.fit_transform(obscene2)
obscene_feature=obscene_feature.toarray()
obscene_feature.shape

obscene_feature_vectorize= vectorizer3.get_feature_names()
obscene_feature_vectorize
obscene_dist = np.sum(obscene_feature, axis=0)
for tag, count in zip(obscene_feature_vectorize,obscene_dist):
    print (tag,count)
obscene_new= pd.DataFrame(obscene_dist)
obscene_new.columns=['word_count']
obscene_new['word'] = pd.Series(obscene_feature_vectorize, index=obscene_new.index)
obscene_new_1=obscene_new[['word','word_count']]
obscene_new_top_15=obscene_new_1.sort_values(['word_count'],ascending=[0])
obscene_new_top_15.head(15)
threat_feature = vectorizer4.fit_transform(threat2)
threat_feature=threat_feature.toarray()
threat_feature.shape

threat_feature_vectorize= vectorizer4.get_feature_names()
threat_feature_vectorize
threat_dist = np.sum(threat_feature, axis=0)
for tag, count in zip(threat_feature_vectorize,threat_dist):
    print (tag,count)
threat_new= pd.DataFrame(threat_dist)
threat_new.columns=['word_count']
threat_new['word'] = pd.Series(threat_feature_vectorize, index=threat_new.index)
threat_new_1=threat_new[['word','word_count']]
threat_new_top_15=threat_new_1.sort_values(['word_count'],ascending=[0])
threat_new_top_15.head(15)
insult_feature = vectorizer5.fit_transform(insult2)
insult_feature=insult_feature.toarray()
insult_feature.shape

insult_feature_vectorize= vectorizer5.get_feature_names()
insult_feature_vectorize

insult_dist = np.sum(insult_feature, axis=0)
for tag, count in zip(insult_feature_vectorize,insult_dist):
    print (tag,count)

insult_new= pd.DataFrame(insult_dist)
insult_new.columns=['word_count']
insult_new['word'] = pd.Series(insult_feature_vectorize, index=insult_new.index)
insult_new_1=insult_new[['word','word_count']]
insult_new_top_15=insult_new_1.sort_values(['word_count'],ascending=[0])
insult_new_top_15.head(15)
identity_hate_feature = vectorizer6.fit_transform(identity_hate2)
identity_hate_feature=identity_hate_feature.toarray()
identity_hate_feature.shape

identity_hate_feature_vectorize= vectorizer6.get_feature_names()
identity_hate_feature_vectorize

identity_hate_dist = np.sum(identity_hate_feature, axis=0)
for tag, count in zip(identity_hate_feature_vectorize,identity_hate_dist):
    print (tag,count)

identity_hate_new= pd.DataFrame(identity_hate_dist)
identity_hate_new.columns=['word_count']
identity_hate_new['word'] = pd.Series(identity_hate_feature_vectorize, index=identity_hate_new.index)
identity_hate_new_1=identity_hate_new[['word','word_count']]
identity_hate_new_top_15=identity_hate_new_1.sort_values(['word_count'],ascending=[0])
identity_hate_new_top_15.head(15)

#time for predction using gaussian naive bayes classifier  
#and  evaliting the train data with confusion_matrix and root mean square error 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x__train = cv.fit_transform(comment_text_train).toarray()
#x__test= cv.fit_transform(comment_text_test).toarray()

y1 = train_data.iloc[:, 2].values
y2 = train_data.iloc[:, 3].values
y3 = train_data.iloc[:, 4].values
y4 = train_data.iloc[:, 5].values
y5 = train_data.iloc[:, 6].values
y6 = train_data.iloc[:, 7].values

from sklearn.cross_validation import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x__train, y1, test_size = 0.40, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(x__train, y2, test_size = 0.40, random_state = 0)
X3_train, X3_test, y3_train, y3_test = train_test_split(x__train, y3, test_size = 0.40, random_state = 0)
X4_train, X4_test, y4_train, y4_test = train_test_split(x__train, y4, test_size = 0.40, random_state = 0)
X5_train, X5_test, y5_train, y5_test = train_test_split(x__train, y5, test_size = 0.40, random_state = 0)
X6_train, X6_test, y6_train, y6_test = train_test_split(x__train, y6, test_size = 0.40, random_state = 0)



from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier2 = GaussianNB()
classifier3 = GaussianNB()
classifier4 = GaussianNB()
classifier5 = GaussianNB()
classifier6 = GaussianNB()

classifier1.fit(X1_train, y1_train)
classifier2.fit(X2_train, y2_train)
classifier3.fit(X3_train, y3_train)
classifier4.fit(X4_train, y4_train)
classifier5.fit(X5_train, y5_train)
classifier6.fit(X6_train, y6_train)

y1_pred = classifier1.predict(X1_test)
y2_pred = classifier2.predict(X2_test)
y3_pred = classifier3.predict(X3_test)
y4_pred = classifier4.predict(X4_test)
y5_pred = classifier5.predict(X5_test)
y6_pred = classifier6.predict(X6_test)
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y1_test, y1_pred)
cm2 = confusion_matrix(y2_test, y2_pred)
cm3 = confusion_matrix(y3_test, y3_pred)
cm4 = confusion_matrix(y4_test, y4_pred)
cm5 = confusion_matrix(y5_test, y5_pred)
cm6 = confusion_matrix(y6_test, y6_pred)
print ("confusion_matrix of toxic is\n:" ,cm1)
print ("confusion_matrix of severe_toxic is\n:" ,cm2)
print ("confusion_matrix of obscene is \n:" ,cm3)
print ("confusion_matrix threat is  \n:" ,cm4)
print ("confusion_matrix of insult is \n:" ,cm5)
print ("confusion_matrix dentity_hate is  \n:" ,cm6)
mse1 = ((y1_pred - y1_test) ** 2).mean()
mse2 = ((y2_pred - y2_test) ** 2).mean()
mse3 = ((y3_pred - y3_test) ** 2).mean()
mse4 = ((y4_pred - y4_test) ** 2).mean()
mse5 = ((y5_pred - y5_test) ** 2).mean()
mse6 = ((y6_pred - y6_test) ** 2).mean()
print("toxic mean square error\n",mse1)
print("severe_toxic mean square error \n",mse2)
print("obscene  mean square error\n",mse3)
print("threat mean square error \n",mse4)
print("insult mean square error \n",mse5)
print("dentity_hate mean square error\n",mse6)
rmse1 = sqrt(mse1)
rmse2 = sqrt(mse2)
rmse3 = sqrt(mse3)
rmse4 = sqrt(mse4)
rmse5 = sqrt(mse5)
rmse6 = sqrt(mse6)


print("toxic  root mean square error \n",rmse1)
print("severe_toxic  root mean square error \n",rmse2)
print("obscene root  mean square error\n",rmse3)
print("threat root  mean square error\n",rmse4)
print("insult root  mean square error \n",rmse5)
print("dentity_hate  root mean square error \n",rmse6)
#last i will upload the test set with the results of my prediction
#thank you all
#resouresecs i use ths https://www.kaggle.com/c/word2vec-nlp-tutorial in kaggle for preapre and clean my data 





