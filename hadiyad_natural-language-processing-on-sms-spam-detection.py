import nltk 
import pandas as pd

df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',delimiter=',',encoding='latin-1')

df.head()
# change the column names;
spam_df=df[['v1','v2']]
spam_df.rename(columns={"v1": "label", "v2": "message"},inplace=True)

spam_df.describe()
spam_df.groupby('label').describe()
spam_df['length'] = spam_df['message'].apply(len)
spam_df.head()
import matplotlib.pyplot as plt
import seaborn as sns

spam_df['length'].plot.hist(bins = 150)
spam_df.describe()
spam_df[spam_df['length']==910]['message'].iloc[0]
spam_df.hist(column='length',by='label',bins=60, figsize=(12,4)) 
import string 
mess = 'Sample message! Notice: it has punctuation.'
string.punctuation
nopunc = [c for c in mess if c not in string.punctuation]
nopunc
from nltk.corpus import stopwords
stopwords.words('english')
nopunc = ''.join(nopunc)
nopunc
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
spam_df.head()
spam_df['message'].head(5).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer
# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(spam_df['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))
message4 = spam_df['message'][3]
print(message4)
bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)
print(bow_transformer.get_feature_names()[3996])
print(bow_transformer.get_feature_names()[9445])
messages_bow = bow_transformer.transform(spam_df['message'])
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
spam_detect_model = MultinomialNB().fit(messages_tfidf, spam_df['label'])
print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', spam_df.label[3])
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)
from sklearn.metrics import classification_report
print (classification_report(spam_df['label'], all_predictions))
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(spam_df['message'], spam_df['label'], test_size=0.2)

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