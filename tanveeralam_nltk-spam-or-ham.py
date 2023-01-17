import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
%matplotlib inline 
raw_data = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',sep=',', encoding='latin-1')
raw_data.head()
raw_data.columns

raw_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
raw_data.head(4)
df = raw_data.copy()
df.info()
df.head(2)
df.tail(2)
df.shape
df.describe()
df.groupby('v1').describe().T
df.rename(columns={"v1":"lable", "v2":"messages"}, inplace=True)
df.head()
df['length'] = df['messages'].apply(len)
df.head()
sns.set_style('darkgrid')
df['length'].plot.hist(bins=100, color='red')
df['length'].describe()
df[df['length'] == 910]['messages'].iloc[0]
sns.set_style('darkgrid')
df.hist(column='length', by='lable', bins=100, figsize=(12,5))
import string
from nltk.corpus import stopwords
def remove_pucn(txt):
    """
    1. remove punctuation 
    2. remove stopwords 
    3. retrun clean text in list format
    """
    no_punc = [t for t in txt if t not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [w for w in no_punc.split() if w.lower() not in stopwords.words('english') ]
df.head(4)
df['messages'].apply(remove_pucn)
from sklearn.feature_extraction.text import CountVectorizer
bag_of_words_transf = CountVectorizer(analyzer=remove_pucn).fit(df['messages'])
bag_of_words_transf.vocabulary_
print(bag_of_words_transf.vocabulary_)
msg50 = df['messages'][49]
print(msg50)
bow50 = bag_of_words_transf.transform([msg50])
print(bow50)
bow50.shape
bag_of_words_transf.get_feature_names()[4777]
features = bag_of_words_transf.get_feature_names()
features = pd.DataFrame(features)
type(features)
features.head(3)
msg_bow = bag_of_words_transf.transform(df['messages'])
msg_bow.shape
## Non Zero Messages
msg_bow.nnz
sparsity = (100.0 * msg_bow.nnz / (msg_bow.shape[0] * msg_bow.shape[1]))
print('sparsity: {}'.format(sparsity))
from sklearn.feature_extraction.text import TfidfTransformer
tf_idf_trans = TfidfTransformer().fit(msg_bow)
tfidf50 = tf_idf_trans.transform(bow50)
print(tfidf50)
tf_idf_trans.idf_[bag_of_words_transf.vocabulary_['actor']]
msg_tf_idf = tf_idf_trans.transform(msg_bow)
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

multiNB = MultinomialNB().fit(msg_tf_idf,df['lable'])

multiNB.predict(tfidf50)
gaussianNB = GaussianNB().fit(msg_tf_idf.toarray(), df['lable'])

gaussianNB.predict(tfidf50.toarray())
bernNB = BernoulliNB().fit(msg_tf_idf, df['lable'])
bernNB.predict(bow50)
multiN_all_pred = multiNB.predict(msg_tf_idf)
gaussian_all_pred = gaussianNB.predict(msg_tf_idf.toarray())
bernNB_all_pred = bernNB.predict(msg_tf_idf)
print('Multinormial Naive Bayes all prediction')
print(multiN_all_pred)
print('\n'*2)

print('Gaussian Naive Bayes all prediction')
print(gaussian_all_pred)
print('\n'*2)

print('Bernoulli Naive Bayes all prediction')
print(bernNB_all_pred)
print('\n'*2)

from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(df['messages'], df['lable'], test_size =.30)
msg_train.shape , msg_test.shape
from sklearn.pipeline import Pipeline
pipeLine = Pipeline([('bow', CountVectorizer(analyzer=remove_pucn)),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', MultinomialNB())
                    ])
pipeLine.fit(msg_train, label_train)
pred = pipeLine.predict(msg_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(label_test, pred))
print(confusion_matrix(label_test, pred))
print(accuracy_score(label_test, pred))