import nltk
# dowloading stopwards package for later nlp use
nltk.download_shell()
# Loding the dataset
messages =[line.rstrip() for line in open('smsspamcollection/SMSSpamCollection',encoding="utf8")]
print(len(messages))
#print the first ten messages and number them using enumerate
for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')
# Convert to a pandas dataframe
import pandas as pd
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()
messages.describe()
messages.groupby('label').describe()
# Creating a new feature length by counting the length of the message
messages['length'] = messages['message'].apply(len)
messages.head()
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
messages['length'].plot(bins=50, kind='hist') 
messages.length.describe()
# Investigating the longest message
messages[messages['length'] == 910]['message'].iloc[0]
# Investigating the pattern betwenn ham and spam message
messages.hist(column='length', by='label', bins=50,figsize=(12,4))
import string
from nltk.corpus import stopwords
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
# Original data frame
messages.head()
# Check to make sure its working
messages['message'].head(5).apply(text_process)
# Show original dataframe
messages.head()
from sklearn.feature_extraction.text import CountVectorizer
# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))
message4 = messages['message'][3]
print(message4)
#Now let's see its vector representation:
bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)
print(bow_transformer.get_feature_names()[4073])
print(bow_transformer.get_feature_names()[9570])
#Now we can use .transform on our Bag-of-Words (bow) transformed object and transform the entire DataFrame of messages. 
messages_bow = bow_transformer.transform(messages['message'])
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
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)
from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

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