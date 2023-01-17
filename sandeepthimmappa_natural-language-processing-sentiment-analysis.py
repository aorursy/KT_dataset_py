amazon = [line.rstrip() for line in open('../input/amazon_cells_labelled.txt')]
for ama_no, mess in enumerate(amazon[:10]):
    print(ama_no, mess)
import pandas as pd
import numpy as np
files = {'amazon': 'amazon_cells_labelled.txt', 
        'yelp':'yelp_labelled.txt', 
        'imdb': 'imdb_labelled.txt'}
df_list = []
for k,v in files.items():
    path = '../input/'+v
    df = pd.read_csv(path, names=['message', 'label'], sep='\t')
    df_list.append(df)
df = pd.concat(df_list, ignore_index=True)
df.head()
df.shape
df['length'] = df['message'].apply(len)
df.head(3)
## importing visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.rcParams['figure.figsize']=(12,5)
plt.rcParams['font.size']=16
plt.style.use('seaborn-whitegrid')
df['length'].plot(kind='hist', bins=100, ec='black')
plt.xlim(0,500)
df['length'].describe()
df[df['length']==479.000000]['message'][2620]
df.hist('length', by='label', ec='black', bins = 100)
df.groupby('label').describe()
## Testing with sample message before proceeding with function
string1 = 'DELETE.. this, film from your mind!'
import string
string.punctuation
##Check for punctuation and join the string
string1_punc = ''.join([i for i in string1.lower() if i not in string.punctuation])
## Importing stopwords from nltk to remove englis stopwords
from nltk.corpus import stopwords
[i for i in string1_punc.split() if i.lower() not in stopwords.words('english')]
## Define function to remove punctuation and stop words

def clean_text(x):
    """
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean words    
    """
    string_punc = ''.join([i for i in x.lower() if i not in string.punctuation])
    string_clean = [i for i in string_punc.split() if i.lower() not in stopwords.words('english')]
    return string_clean
df['message'].head(5).apply(clean_text)
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(analyzer=clean_text).fit(df['message'])
## Will transform just one text message

mess2 = df['message'][1]
cv2 = count_vector.transform([mess2])
print(cv2)
print(cv2.shape)
count_vector.get_feature_names()[1639]
df_bow = count_vector.transform(df['message'])
print('shape of sparse matrix:{}'.format(df_bow.shape))
print('Number of non zero occurance:{}'.format(df_bow.nnz))
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer().fit(df_bow)
## Transforming count vector of mess2
tfidf2 = tfidf.transform(cv2)
print(tfidf2)
## Checking IDF of word 'good'
tfidf.idf_[count_vector.vocabulary_['good']]
df_tfidf = tfidf.transform(df_bow)
print(df_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB().fit(df_tfidf, df['label'])
print(nb.predict(tfidf2)[0])
print(df['label'][1])
nb_predict = nb.predict(df_tfidf)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(df['label'], nb_predict))
print('\n')
print(confusion_matrix(df['label'], nb_predict))
from sklearn.model_selection import train_test_split
message_train, message_test, label_train, label_test = train_test_split(df['message'], df['label'], test_size=0.3, random_state=101)
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('bow', CountVectorizer()),
         ('tfidf', TfidfTransformer()),
         ('classifier', MultinomialNB())])
pipeline.fit(message_train, label_train)
final_predict = pipeline.predict(message_test)
print(classification_report(label_test, final_predict))
print('\n')
print(confusion_matrix(label_test, final_predict))