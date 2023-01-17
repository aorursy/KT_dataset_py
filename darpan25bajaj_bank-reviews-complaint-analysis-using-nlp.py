import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import string

import nltk

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer

from textblob import TextBlob

from wordcloud import WordCloud, STOPWORDS

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

import nltk

from nltk.corpus import wordnet
customer = pd.read_csv('../input/bank-reviewcomplaint-analysis/BankReviews.csv', encoding='windows-1252' )

customer.head()
customer.info()
customer.shape
customer.isnull().sum()
customer['Stars'].value_counts()
plt.figure(figsize=(8,6))

sns.countplot(customer.Stars)

plt.show()
X = customer['Reviews']

Y = customer['Stars']
X.head()
# UDF to find sentiment polarity of the reviews

def sentiment_review(text):

    analysis = TextBlob(text)

    polarity_text = analysis.sentiment.polarity

    if polarity_text > 0:

        return 'Positive'

    elif polarity_text == 0:

        return 'Neutral'

    else:

        return 'Negative'  
# creating dictionary which will contain both the review and the sentiment of the review

final_dictionary = []

for text in X:

    dictionary_sentiment = {}

    dictionary_sentiment['Review'] = text

    dictionary_sentiment['Sentiment'] = sentiment_review(text)

    final_dictionary.append(dictionary_sentiment)

print(final_dictionary[:5])
# Finding positive reviews

positive_reviews = []

for review in final_dictionary:

    if review['Sentiment'] =='Positive':

        positive_reviews.append(review)

print(positive_reviews[:5])

    
# Finding neutral reviews

neutral_reviews = []

for review in final_dictionary:

    if review['Sentiment'] =='Neutral':

        neutral_reviews.append(review)

print(neutral_reviews[:5])
# Finding negative reviews

negative_reviews = []

for review in final_dictionary:

    if review['Sentiment'] =='Negative':

        negative_reviews.append(review)

print(negative_reviews[:5])
# counting number of positive,neutral and negative reviews

reviews_count = pd.DataFrame([len(positive_reviews),len(neutral_reviews),len(negative_reviews)],index=['Positive','Neutral','Negative'])
reviews_count
reviews_count.plot(kind='bar')

plt.ylabel('Reviews Count')   

plt.show()
# printing first five positive reviews

i=1

for review in positive_reviews[:5]:

        print(i)

        print(review['Review'])

        print('******************************************************')

        i+=1
# printing first five negative reviews

i=1

for review in negative_reviews[:5]:

        print(i)

        print(review['Review'])

        print('******************************************************')

        i+=1
# UDF to clean the reviews

def clean_text(text):

    text = text.lower()

    text = text.strip()

    text = "".join([char for char in text if char not in string.punctuation])

    return text
# X = customer['Reviews']

X.head()
# applying clean_text function defined above to remove punctuation, strip extra spaces and convert each word to lowercase

X = X.apply(lambda y: clean_text(y))
X.head()
tokens_vect = CountVectorizer(stop_words='english')
token_dtm = tokens_vect.fit_transform(X)
tokens_vect.get_feature_names()
token_dtm.toarray()
token_dtm.toarray().shape
len(tokens_vect.get_feature_names())
pd.DataFrame(token_dtm.toarray(),columns = tokens_vect.get_feature_names())
print(token_dtm)
# creating a dataframe which shows the count of how many times a word is coming in the corpus

count_dtm_dataframe = pd.DataFrame(np.sum(token_dtm.toarray(),axis=0),tokens_vect.get_feature_names()).reset_index()

count_dtm_dataframe.columns =['Word','Count']
count_dtm_dataframe.head()
#adding sentiment column which shows sentiment polarity of each word

sentiment_word = []

for word in count_dtm_dataframe['Word']:

    sentiment_word.append(sentiment_review(word))

count_dtm_dataframe['Sentiment'] = sentiment_word
count_dtm_dataframe.head()
# separating positive words

positive_words_df= count_dtm_dataframe.loc[count_dtm_dataframe['Sentiment']=='Positive',:].sort_values('Count',ascending=False)
positive_words_df.head(20)
# plotting word cloud of 10 most frequently used positive words

wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(positive_words_df.iloc[0:11,0]))

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
# separating negative words

negative_words_df= count_dtm_dataframe.loc[count_dtm_dataframe['Sentiment']=='Negative',:].sort_values('Count',ascending=False)
negative_words_df.head(10)
# plotting word cloud of 10 most frequently used positive words

wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(negative_words_df.iloc[0:11,0]))

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,random_state = 123, test_size = 0.2)  
print('No.of observations in train_X: ',len(train_X), '| No.of observations in test_X: ',len(test_X))

print('No.of observations in train_Y: ',len(train_Y), '| No.of observations in test_Y: ',len(test_Y))
vect = CountVectorizer(strip_accents='unicode', stop_words='english', ngram_range=(1,1),min_df=0.001,max_df=0.95)
train_X_fit = vect.fit(train_X)

train_X_dtm = vect.transform(train_X)

test_X_dtm = vect.transform(test_X)
print(train_X_dtm)
print(test_X_dtm)
vect.get_feature_names()
print('No.of features for are',len(vect.get_feature_names()))
train_X_dtm_df = pd.DataFrame(train_X_dtm.toarray(),columns=vect.get_feature_names())
train_X_dtm_df.head()
# Finding how many times a tem is used in corpus

train_dtm_freq = np.sum(train_X_dtm_df,axis=0)
train_dtm_freq.head(20)
vect_tdm = TfidfVectorizer(strip_accents='unicode', stop_words='english', ngram_range=(1,1),min_df=0.001,max_df=0.95)
train_X_tdm = vect_tdm.fit_transform(train_X)

test_X_tdm = vect.transform(test_X)
print(train_X_tdm)
print(test_X_tdm)
vect_tdm.get_feature_names()
print('No.of features for are',len(vect_tdm.get_feature_names()))
# creating dataframe to to see which features are present in the documents

train_X_tdm_df = pd.DataFrame(train_X_tdm.toarray(),columns=vect_tdm.get_feature_names())
train_X_tdm_df.head()
test_X_tdm_df = pd.DataFrame(test_X_tdm.toarray(),columns=vect_tdm.get_feature_names())
test_X_tdm_df.head()
# Finding how many times a term is used in test corpus

test_tdm_freq = np.sum(test_X_tdm_df,axis=0)
test_tdm_freq.head(20)
# train a LDA Model

lda_model = LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=50)

X_topics = lda_model.fit_transform(train_X_tdm)

topic_word = lda_model.components_ 

vocab = vect.get_feature_names()
# view the topic models

top_words = 10

topic_summaries = []

for i, topic_dist in enumerate(topic_word):

    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(top_words+1):-1]

    topic_summaries.append(' '.join(topic_words))

    print(topic_words)
# view the topic models

top_words = 10

topic_summaries = []

for i, topic_dist in enumerate(topic_word):

    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(top_words+1):-1]

    topic_summaries.append(' '.join(topic_words))

topic_summaries
# building naive bayes model on DTM

naive_model = MultinomialNB()

naive_model.fit(train_X_dtm,train_Y)
predict_train = naive_model.predict(train_X_dtm)

predict_test = naive_model.predict(test_X_dtm)
len(predict_test)
print('Accuracy on train: ',metrics.accuracy_score(train_Y,predict_train))

print('Accuracy on test: ',metrics.accuracy_score(test_Y,predict_test))
# predict probabilities on train and test

predict_prob_train = naive_model.predict_proba(train_X_dtm)[:,1]

predict_prob_test = naive_model.predict_proba(test_X_dtm)[:,1]
print('ROC_AUC score on train: ',metrics.roc_auc_score(train_Y,predict_prob_train))

print('ROC_AUC score on test: ',metrics.roc_auc_score(test_Y,predict_prob_test))
# confusion matrix on test 

cm_test = metrics.confusion_matrix(test_Y,predict_test,[5,1])
cm_test
import seaborn as sns

sns.heatmap(cm_test,annot=True,xticklabels=[5,1],yticklabels=[5,1])

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()
# building naive bayes model on DTM

naive_model = MultinomialNB()

naive_model.fit(train_X_tdm,train_Y)
predict_train = naive_model.predict(train_X_tdm)

predict_test = naive_model.predict(test_X_tdm)
len(predict_test)
print('Accuracy on train: ',metrics.accuracy_score(train_Y,predict_train))

print('Accuracy on test: ',metrics.accuracy_score(test_Y,predict_test))
# predict probabilities on train and test

predict_prob_train = naive_model.predict_proba(train_X_tdm)[:,1]

predict_prob_test = naive_model.predict_proba(test_X_tdm)[:,1]
print('ROC_AUC score on train: ',metrics.roc_auc_score(train_Y,predict_prob_train))

print('ROC_AUC score on test: ',metrics.roc_auc_score(test_Y,predict_prob_test))
# confusion matrix on test 

cm_test = metrics.confusion_matrix(test_Y,predict_test,[5,1])
cm_test
import seaborn as sns

sns.heatmap(cm_test,annot=True,xticklabels=[5,1],yticklabels=[5,1])

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()