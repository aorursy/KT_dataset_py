import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
X_text = []
Y = [ ]
with open('../input/source_task_train.csv','r') as f:
    print(f.readline())
    for line in f:
        data = line.split(',')
        X_text.append(data[1])
        Y.append(data[2])
Y = np.array(Y)
count_vectorizer = CountVectorizer(max_df=0.9, min_df=2)
count_vectorizer.fit(X_text)
X = count_vectorizer.transform(X_text)
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
X.shape
_pca = TruncatedSVD(n_components=40)
X_pca =  _pca.fit_transform(X)
X
from scipy.sparse import csr_matrix, hstack, vstack
csr_matrix(X_pca)
hstack([X, X_pca])
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
lg = LogisticRegression(
    penalty='l2',
    C=10, 
    n_jobs=-1, verbose=1, 
    
    solver='sag', multi_class='multinomial',
    
    max_iter=300
)
lg.fit(X_train, Y_train)
Y_pred = lg.predict(X_test)
print(classification_report(Y_test, Y_pred, digits=6))
confusion_matrix(Y_test, Y_pred)
# Достаём из векторайзера словарь
vocab = count_vectorizer.vocabulary_.items()
vocab = sorted(list(vocab), key=lambda x: x[1])
vocab_words, vocab_index = zip(*vocab)
vocab_words = np.array(vocab_words)
# vocab_words[-100:]
for label in range(4):
    _class_coef = lg.coef_[label]
    print('Class', label, 'слова увеличивающие вероятность класса:')
    print(list(vocab_words[ (-_class_coef).argsort()][:100]))
    print()
    print('Class', label,  'слова уменьшающие вероятность класса:')
    print(list(vocab_words[ (_class_coef).argsort()][:100]))
    print('-'*80)
X_target = []
_Id = []
with open('../input/source_task_test_without_labels.csv','r') as f:
    print(f.readline())
    for line in f:
        data = line.split(',')
        X_target.append(data[1])
        _Id.append(data[0])
X_ = count_vectorizer.transform(X_target)
Y_target = lg.predict(X_)
with open('count_vec_lg.csv', 'w') as f:
    f.write('_id,label\n')
    for _id, y  in zip(_Id, Y_target):
        f.write('%s,%s\n' % (_id, y))

tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
X = tfidf_vectorizer.fit_transform(X_text)
X.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
lg = LogisticRegression(
    penalty='l2',
    C=50, 
    n_jobs=-1, verbose=1, 
    solver='sag', multi_class='multinomial',
    max_iter=300
)
lg.fit(X_train, Y_train)
Y_pred = lg.predict(X_test)
print(classification_report(Y_test, Y_pred, digits=6))
# Достаём из векторайзера словарь
vocab = tfidf_vectorizer.vocabulary_.items()
vocab = sorted(list(vocab), key=lambda x: x[1])
vocab_words, vocab_index = zip(*vocab)
vocab_words = np.array(vocab_words)
for label in range(4):
    _class_coef = lg.coef_[label]
    print('Class', label, 'слова увеличивающие вероятность класса:')
    print(list(vocab_words[ (-_class_coef).argsort()][:100]))
    print()
    print('Class', label,  'слова уменьшающие вероятность класса:')
    print(list(vocab_words[ (_class_coef).argsort()][:100]))
    print('-'*80)
X_= tfidf_vectorizer.transform(X_target)
Y_target = lg.predict(X_)
with open('tfidf_vec_lg.csv', 'w') as f:
    f.write('_id,label\n')
    for _id, y  in zip(_Id, Y_target):
        f.write('%s,%s\n' % (_id, y))

from sklearn.ensemble import RandomForestClassifier
clrTree = RandomForestClassifier(
    n_estimators=100,
    random_state=1,
    max_depth=10,
    min_samples_leaf=1,
    verbose=1,
)
clrTree = clrTree.fit(X_train, Y_train)
outTree = clrTree.predict(X_)
with open('rf.csv', 'w') as f:
    f.write('_id,label\n')
    for _id, y  in zip(_Id, outTree):
        f.write('%s,%s\n' % (_id, y))
# !pip install pymorphy2
# import pymorphy2
# morph = pymorphy2.MorphAnalyzer()
# parsed_word = morph.parse('борща')[0]
# parsed_word
# parsed_word.normal_form
# parsed_word.tag
# parsed_word.tag.POS
# morph.parse('стали')
