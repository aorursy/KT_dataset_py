import nltk
message =[line.rstrip() for line in open('../input/SMSSpamCollection')]
print(len(message))
for message_no, message in enumerate(message[:10]):
    print(message_no, message)
    print('\n')
import pandas as pd
message = pd.read_csv('../input/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
message.head()
message.describe()
message.groupby('label').describe()
message['length'] = message['message'].apply(len)
message.head()
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline
message['length'].plot(bins=50, kind='hist') 
message.length.describe()
message[message['length'] == 910]['message'].iloc[0]
message.hist(column='length', by='label', bins=50,figsize=(12,4))
import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)
from nltk.corpus import stopwords
stopwords.words('english')[0:10] # Show some stop words
nopunc.split()
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess
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
message.head()
# Check to make sure its working
message['message'].head(5).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer
# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(message['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))
message4 = message['message'][3]
print(message4)
bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)
print(bow_transformer.get_feature_names()[4073])
print(bow_transformer.get_feature_names()[9570])
messages_bow = bow_transformer.transform(message['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, message['label'])
print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', message.label[3])
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)
from sklearn.metrics import classification_report
print (classification_report(message['label'], all_predictions))
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(message['message'], message['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))
from nltk.corpus import stopwords

#doing the stamming that is taking only root of the word that is loved can be taken as love
from nltk.stem.porter import PorterStemmer
#corpus is collection of text
import re
corpus =[]
for i in range(0 ,5572):
    review = re.sub('[^a-zA-Z]', ' ' ,message['message'][i])
    review =review.lower()
    review =review.split()
    review =[word for  word in review if not word in set(stopwords.words('english'))]
    portstemmer =PorterStemmer()
    review =[portstemmer.stem(word) for  word in review if not word in set(stopwords.words('english'))]
    review =' '.join(review)
    corpus.append(review)
corpus

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = message.iloc[:, 1].values
X
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of meaasge')

plt.legend()
plt.show()
