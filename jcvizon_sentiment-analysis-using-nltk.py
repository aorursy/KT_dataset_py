'''
Import all the necessary libraries
'''
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud.wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.style
import pandas as pd
import string

%matplotlib inline

'''
Importing Raw data
'''
text=open('../input/responses/ref.txt',"r")
raw_data=text.read()
print(raw_data)
'''
Stopwords is a built-in data set of NLTK that deemed irrelevant for searching purposes because they occur frequently in the language for which the indexing engine has been tuned
'''
print(stopwords.words('english'))
'''
We need to assign the stopwords to a variable which soon be used to filter our raw_data
'''
eng_sw=stopwords.words('english')
'''
We also need to tokenize the raw data or basically split them into a list/array.
'''
text_array=word_tokenize(raw_data)
'''
Now we are going to filter the text_array with these 2 steps:
1. Filter the tokenize raw_data using eng_sw
2. Filter the filtered data using nltk.words which is compose of legit english words. (removing misspelled word)
'''
eng_words=set(nltk.corpus.words.words())
sw_filter_text=[item for item in text_array if item not in eng_sw]
eng_filter_text=[word for word in sw_filter_text if word in eng_words]
'''
Creating Positive and Negative List
'''
PText=open('../input/positvenegative-words/positive.txt','r')
positive=PText.read().split()

NText=open('../input/positvenegative-words/negative.txt','r')
negative=NText.read().split()
print('Sample of positive + :',positive[0:20])
print('Sample of negative - :',negative[0:20])
'''
Let us now use get the positive and negative data from our eng_filter_text
'''
filter_positive=[word for word in eng_filter_text if word in positive]
filter_negative=[word for word in eng_filter_text if word in negative]
'''
Let us now count the frequency for each text array:
1.raw_data
2.filter_positive
3.filter_negative
'''
count_raw_data=Counter(raw_data)
count_filter_positive=Counter(filter_positive)
count_filter_negative=Counter(filter_negative)
print('Raw Data:', count_raw_data)
print('Positive:', count_filter_positive)
print('Negative:', count_filter_negative)
'''
Now we have to convert our filtered data and remove duplicated words for plotting purposes. 
'''
final_text=' '.join(eng_filter_text)
final_positive_text=' '.join(filter_positive)
final_negative_text=' '.join(filter_negative)
plt.figure(figsize=(20,10))
wordcloud=WordCloud(background_color='white',mode='RGB',width=1800,height=1400,relative_scaling=0,prefer_horizontal=0.5).generate(final_text)
plt.title('Raw-Text WordCloud')
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
plt.figure(figsize=(20,10))
wordcloud=WordCloud(background_color='white',mode='RGB',width=1800,height=1400,relative_scaling=0,prefer_horizontal=0.5).generate(final_positive_text)
plt.title('Positive-Text WordCloud')
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
plt.figure(figsize=(20,10))
wordcloud=WordCloud(background_color='white',mode='RGB',width=1800,height=1400,relative_scaling=0,prefer_horizontal=0.5).generate(final_negative_text)
plt.title('Negative-Text WordCloud')
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
final_text_list=raw_data.split('\n')
senti_analyzer=SentimentIntensityAnalyzer()
for sentence in final_text_list:
    print(sentence)
    analyzer = senti_analyzer.polarity_scores(sentence)
    for k in sorted(analyzer):
        print('{0}: {1}, '.format(k, analyzer[k]), end='\n')
        
        
'''
Function to return either Positive or Negative
'''
def polarity(sentence):
    for sen in sentence:
        analyzer = senti_analyzer.polarity_scores(sentence)
        if analyzer['neg']>analyzer['pos']:
            return ('Negative')
        elif analyzer['pos']>analyzer['neg']:
            return('Positive')
        else:
            pass
        
'''
Create new dataframe consist of raw data and polarity result
'''
new_df=pd.DataFrame({"Text":final_text_list,"Result":''})
'''
Application of Polarity function
'''
new_df['Result']=new_df['Text'].apply(polarity)
new_df.head()
new_df.shape
'''
Delete rows consist of missing data
'''
new_df.dropna(axis=0,inplace=True)
new_df.head()
from sklearn.feature_extraction.text import CountVectorizer
'''
Bag of words / Word2Vec
'''
bow_transformer=CountVectorizer()
bow_transformer.fit(new_df['Text'])
print(bow_transformer.vocabulary_)
sparse_matrix=bow_transformer.transform(new_df['Text'])
print('Shape of Sparse Matrix: ', sparse_matrix.shape)
print('Amount of Non-Zero occurences: ', sparse_matrix.nnz)
sparsity = (100.0 * sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]))
print('sparsity: {}'.format(sparsity))
from sklearn.feature_extraction.text import TfidfTransformer
'''
Term Frequency Inverse Document Frequency 
https://en.wikipedia.org/wiki/Tf%E2%80%93idf
'''

tfidf_transformer=TfidfTransformer()
tfidf=tfidf_transformer.fit_transform(sparse_matrix)
print('TFIDF Sample Results')
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['learned']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['violate']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['urgent']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['responsible']])

comments_tfidf = tfidf_transformer.transform(sparse_matrix)
print(comments_tfidf)
'''
Import Multinomial Naive Bayes
'''
from sklearn.naive_bayes import MultinomialNB
'''
Create a model defining the feature: comments_tfidf, and target:df['Result']
'''
polar_prediction_model=MultinomialNB().fit(comments_tfidf,new_df['Result'])
tfidf.shape

print('Sample Predictions:')
print('predicted:', polar_prediction_model.predict(tfidf)[0],'|',polar_prediction_model.predict(tfidf)[100])
print('expected:', new_df.Result[0],'|',new_df.Result[100])
'''
The model generates predictions
'''
all_predictions=polar_prediction_model.predict(comments_tfidf)
print(all_predictions)
'''
Classification Report to measure the accuracy of the created model
'''
from sklearn.metrics import classification_report
print(classification_report(new_df['Result'],all_predictions))
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(new_df['Result'],all_predictions))
print(accuracy_score(new_df['Result'],all_predictions))
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
test_text=open('../input/reviews-for-testing/Reflection1.txt',"r")
test_raw_data=test_text.read()
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(new_df['Text'],new_df['Result'])
to_predict=test_raw_data.split('\n')
test_predictions=pipeline.predict(to_predict)
to_predict[39]
test_predictions[39]

