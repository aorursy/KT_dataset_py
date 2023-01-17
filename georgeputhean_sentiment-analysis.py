import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
tweet_df=pd.read_csv("../input/twitter-sentiment-analysis-hatred-speech/train.csv")
tweet_df.head(10)
tweet_df.info()
tweet_df.describe()
tweet_df['tweet'].head(10)
tweet_df=tweet_df.drop(columns=['id'])
tweet_df.head(10)
tweet=tweet_df['tweet']
label=tweet_df['label']
plt.hist(label,bins=10,edgecolor='black')
sns.heatmap(tweet_df.isnull(), 
            fmt="d", 
            linewidths=.5, 
            cmap='Blues');
sns.countplot(x='label',data=tweet_df)
tweet_df['length']=tweet_df['tweet'].apply(len)
tweet_df.head(2)
plt.hist(x=tweet_df['length'],bins=100)
plt.show()
tweet_df.describe()
tweet_df[tweet_df['length']==84]['tweet'].iloc[0:100]
    
positive=tweet_df[tweet_df['label']==0]
negative=tweet_df[tweet_df['label']==1]
negative.head()
tweet_df['tweet'].iloc[13]
from wordcloud import WordCloud
tweet_sentence=" ".join(tweet)
plt.imshow(WordCloud().generate(tweet_sentence))
plt.figure(figsize=(50,40))
negative_sentence=negative['tweet'].tolist()
negative_sentence=" ".join(negative_sentence)
plt.imshow(WordCloud().generate(negative_sentence))
plt.figure(figsize=(50,40))
#remove the punctuation
import string
string.punctuation
tweet_rem=[]
for line in tweet:
    line=''.join([char for char in line if char not in string.punctuation])
    tweet_rem.append(line)
tweet_rem
tweet_df['no_punc']=tweet_rem
tweet_df
from nltk.corpus import stopwords
stopwords.words('english')
twitter_nostop=[]
for line in tweet_df['no_punc']:
    twitter_stop=' '.join([ i for i in line.split() if i.lower() not in stopwords.words('english')])
    twitter_nostop.append(twitter_stop)
twitter_nostop
tweet_df['no_stop_punc']=twitter_nostop
tweet_df
from sklearn.feature_extraction.text import CountVectorizer
sample=["Hi this is py monkey to soon","monkey lays for a reporting", "reporting!! can you come back to me soon"]
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(sample)
print(x.toarray())
print(vectorizer.get_feature_names())
def message_cleaning(message):
    test_punc_removed=[char for char in message if char not in  string.punctuation]
    test_punc_removed_join=''.join(test_punc_removed)
    test_punc_stop_removed=[word for word in test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return test_punc_stop_removed

tweet_df['clean_tweet']=tweet_df['tweet'].apply(message_cleaning)
tweet_df['tweet'][5]
tweetvectorizer=CountVectorizer(analyzer=message_cleaning).fit_transform(tweet_df['tweet']).toarray()
tweetvectorizer.shape
x=tweetvectorizer
y=tweet_df['label']
y=tweet_df['label']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,yp_test = train_test_split(x,y,test_size=0.3)
from sklearn.naive_bayes import MultinomialNB
NB_classifier=MultinomialNB()
NB_classifier.fit(x_train,y_train)
ypred_test=NB_classifier.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix

cm=confusion_matrix(yp_test,ypred_test)
sns.heatmap(cm,annot=True)
print(classification_report(yp_test,ypred_test))
 