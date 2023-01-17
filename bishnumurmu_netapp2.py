import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_json('../input/training_dataset/dataset.json')
train.info()
X = train.drop('category',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, train['category'], test_size=0.5, random_state=101,shuffle = False)
X.headline = X.headline.apply(lambda x: x.lower())
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt
X['headline'] = np.vectorize(remove_pattern)(X['headline'], "@[\w]*")
X['headline'] = X['headline'].str.replace("[^a-zA-Z#']", " ")
X.head()
X['headline'] = X['headline'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>4]))
X.head()
tokenized_tweet = X['headline'].apply(lambda x: x.split())
tokenized_tweet.head()
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
X['headline'] = tokenized_tweet
'''all_words = ' '.join([text for text in X['tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()'''
'''combi = X.append(train['label'], ignore_index=True)'''
'''import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import string
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess]
    
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english') + ['http','www','https']]'''
'''vectorizer = CountVectorizer(analyzer=text_process)'''
'''sparsed_data = vectorizer.fit_transform(X['headline'])
sparsed_data'''
'''print ('Shape of Sparse Matrix: ', sparsed_data.shape)
print ('Amount of Non-Zero occurences: ', sparsed_data.nnz)
print ('sparsity: %.2f%%' % (100.0 * sparsed_data.nnz /
                             (sparsed_data.shape[0] * sparsed_data.shape[1])))'''
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2,stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(X['headline'])


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(bow, train['category'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict(xvalid_bow)

accuracy_score(yvalid, prediction) # calculating f1 score
'''nx_train = messages_tfidf[0:100426]
nx_test = messages_tfidf[100426:200853]'''
'''from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
lreg = LogisticRegression()'''
'''logr.fit(nx_train,y_train)'''
