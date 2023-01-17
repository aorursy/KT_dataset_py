# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

FILEPATH = '/kaggle/input/sms-spam-collection-dataset/spam.csv'
df = pd.read_csv(FILEPATH, encoding='iso-8859-1', engine = 'c') # engine 'c' used instead of 'python' for higher performance
df.sample(10)
# delete unnecessary cols
cols = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']

df.drop(cols, axis = 1, inplace = True)
df.head()
# Title change v1 = result, v2 = input

df.columns = ['result', 'input']

# we can also use df.rename() option here
df.head()
# reorder options - must be applicable for all cols
df = df[['input','result']]
 
df.head()
# Rename cols by using .rename - can be used for selected cols

df.rename(columns = {'input' : 'my_new_input', 'result' : 'my_new_result'}, inplace = True)
df.head()
df.count()
# print first string

df.iloc[1]
df.iloc[2][1]
def find_message_length(msg):
    
    msg_words = msg.split(' ')
    
    msg_len = len(msg_words)
    
    return msg_len
print(find_message_length('spaCy is an open-source software library for advanced natural language processing, written in the programming languages Python and Cython. The library is published under the MIT license and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion.'))
# Create a new col called 'message_word_length' showing how many words in the message
df['input_words_count'] = df['my_new_input'].apply(find_message_length)
df.head()

# ref: https://rajacsp.github.io/mlnotes/python/data-wrangling/advanced-custom-lambda/
# show the unique labels

set(df['my_new_result'])
def find_length(msg):
    
    msg_len = len(msg)
    
    return msg_len
print(find_length(df.iloc[0][0]))
# Create a new col called 'message_word_length' showing how many words in the message
df['input_char_length'] = df['my_new_input'].apply(find_length)
df.head()
# History words count

import matplotlib.pyplot as plt

# to avoid popups use inline
%matplotlib inline 
# plt.hist(data['label'], bins=3, weights=np.ones(len(data['label'])) / len(data['label']))

import numpy as np

plt.hist(df['input_words_count'], bins = 100, weights = np.ones(len(df['input_words_count'])) / len(df['input_words_count']))

plt.xlabel('Word Length')
plt.ylabel('Group Count')
plt.title('Word Length Histogram')
# Find more than 80 words
df['input_words_count']
df_above_80 = df[df['input_words_count'] > 80]
df_above_80.sort_values(by='input_words_count')
import numpy as np

plt.hist(df['input_char_length'], bins = 100, weights = np.ones(len(df['input_char_length'])) / len(df['input_char_length']))

plt.xlabel('Char Length')
plt.ylabel('Group Count')
plt.title('Char Length Histogram')
df.my_new_result.value_counts()
spams = df[df['my_new_result']=='spam'].iloc[: ,0]
spams[:5]
hams = df[df['my_new_result']=='ham'].iloc[:,0]
hams[:5]
plt.hist(spams.apply(lambda msg : len(msg)),bins = 100,label = 'Spams')

plt.hist(hams.apply(lambda msg : len(msg)),bins=100,label='Hams',alpha=0.3)

plt.xlabel('Message length')

plt.ylabel('Count')

plt.title('String lengths')

plt.legend()

plt.show()
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en')
def normalize(msg):
    doc = nlp(msg)
    res=[]
    
    for token in doc:
        if(token.is_stop or token.is_digit or token.is_punct or not(token.is_oov)):
            pass
        else:
            res.append(token.lemma_.lower())
    return res
normalize('spaCy is an open-source software library for advanced natural language processing, written in the programming languages Python and Cython. The library is published under the MIT license and its 23 main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion.')
spam_tokens = []
for spam in spams:
    spam_tokens += normalize(spam)
    
ham_tokens = []
for ham in hams:
    ham_tokens += normalize(ham)
from collections import Counter
most_common_tokens_in_spams = Counter(spam_tokens).most_common(20)
most_common_tokens_in_hams = Counter(ham_tokens).most_common(20)

print(most_common_tokens_in_spams,end="\n\n")
print(most_common_tokens_in_hams)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
binary_vectorizer = CountVectorizer(binary=True)
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()
def feature_extraction(msg):
    
    mat = pd.DataFrame(tfidf_vectorizer.fit_transform(msg).toarray(),columns=tfidf_vectorizer.get_feature_names(),index=None)
    return mat
from sklearn.model_selection import train_test_split
df['my_new_result']=df['my_new_result'].map({"ham":0,"spam":1})
k=feature_extraction(df['my_new_input'])
print(k.shape,df['my_new_result'].shape)
train_x,train_y, test_x,test_y = train_test_split(k,df['my_new_result'], test_size=0.3)

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
clfs = {
    'mnb': MultinomialNB(),
    'gnb': GaussianNB(),
    'svm1': SVC(kernel='linear'),
    'svm2': SVC(kernel='rbf'),
    'svm3': SVC(kernel='sigmoid'),
    'mlp1': MLPClassifier(),
    'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),
    'ada': AdaBoostClassifier(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'gbc': GradientBoostingClassifier(),
    'lr': LogisticRegression()
}
f1_scores = dict()
for clf_name in clfs:
    clf = clfs[clf_name]
    clf.fit(train_x, test_x)
    y_pred = clf.predict(train_y)
    f1_scores[clf_name] = f1_score(y_pred, test_y)
    print(clf_name, f1_scores[clf_name])
solver=['lbfgs', 'sgd', 'adam']
max_f1_score = float('-inf')
best_solver = None

for s in solver:
    
    clf = MLPClassifier(solver = s)
    clf.fit(train_x, test_x)
    y_pred = clf.predict(train_y)
    current_f1_score = f1_score(y_pred, test_y)
    if current_f1_score > max_f1_score:
        max_f1_score = current_f1_score
        best_solver = s
        
print('Best Solver: {0}'.format(best_solver))
alpha_values = [i * 0.1 for i in range(11)]
max_f1_score = float('-inf')
best_alpha = None
for alpha in alpha_values:
    clf = MLPClassifier(solver = 'adam')
    clf.fit(train_x, test_x)
    y_pred = clf.predict(train_y)
    current_f1_score = f1_score(y_pred, test_y)
    if current_f1_score > max_f1_score:
        max_f1_score = current_f1_score
        best_alpha = alpha
        
print('Best f1-score: {0}'.format(max_f1_score))
print('Best alpha: {0}'.format(best_alpha))
clf = MLPClassifier(solver = 'lbfgs', alpha=0.4)
clf.fit(train_x, test_x)
y_pred = clf.predict(train_y)
print(confusion_matrix(y_pred, test_y))
import seaborn as sns
sns.regplot(x=test_y,y=y_pred,marker="*")