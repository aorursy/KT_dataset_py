# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import SGD
import seaborn as sns
data_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
data_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
y_train_data = data_train["target"]
data_train.head()
data_test.head()
data_train.shape
data_train.isna().sum()
data_train.head()
def eliminate_symbols(text):
    """
    this function allows to keep only letters in the text
    """
    text =re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub("[^a-zA-^Z]", " ", text)
    
    return text

data_train["text"] = data_train["text"].apply(lambda x : eliminate_symbols(x))
#Text to lowercase
data_train["text"] = data_train["text"].apply(lambda x : x.lower())
# Tokenize text
data_train["text"]=data_train["text"].apply(lambda x : word_tokenize(x))
#Eliminate stopwords
def eliminate_stop_words(token):
    new=[]
    for word in token:
        if word not in stopwords.words('english'):
            new.append(word)
    return new

data_train["text"]=data_train["text"].apply(lambda x : eliminate_stop_words(x))

def token_toString(token):
    """
    After transforming sentenses to token we need to return them as sentences
    """
    text=""
    for j in token:
        text+=" "+j
    return text
data_train["text"]=data_train["text"].apply(lambda x : token_toString(x))

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(data_train['text'], 20)
df2 = pd.DataFrame(common_words, columns = ['text' , 'count'])
df2.groupby('text').sum()['count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 words')

def get_top_bi_words(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_bi_words(data_train['text'], 20)
df2 = pd.DataFrame(common_words, columns = ['text' , 'count'])
df2.groupby('text').sum()['count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 Bigrams')
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=data_train[data_train['target']==1]['text'].str.len()
ax1.hist(tweet_len,color='red')
ax1.set_title('disaster tweets')
tweet_len=data_train[data_train['target']==0]['text'].str.len()
ax2.hist(tweet_len,color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.show()
nlp = spacy.load('en_core_web_lg')
train_vectors = np.array([nlp(text).vector for text in data_train['text']])
test_vectors = np.array([nlp(text).vector for text in data_test['text']])
# Here is a little example to see how CountVectorizer works
# it takes all the vocabulary in the dataset and make each word as feature the it puts zeros and ones the feature for each sentence
# in the dataset
count_vect = CountVectorizer()
X_data_count = count_vect.fit_transform(["when i saw your shadow","I am swimming am and the waves kicked me out"])
print(pd.DataFrame(X_data_count.A, columns=count_vect.get_feature_names()).to_string())
count_vect = CountVectorizer()
count_vect.fit(data_train['text'])

train_vectors1 = count_vect.transform(data_train['text'])
test_vectors1 = count_vect.transform(data_test['text'])
tfidf_transformer = TfidfTransformer()
X_data_train_tfidf = tfidf_transformer.fit_transform(train_vectors1)
X_data_test_tfidf = tfidf_transformer.transform(test_vectors1)
X_train_spacy_vectorizer, X_test_spacy_vectorizer, y_train_spacy_vectorizer, y_test_spacy_vectorizer = train_test_split(train_vectors, y_train_data, test_size= 0.2, shuffle= True, random_state= 42)
X_train_count_vectorizer, X_test_count_vectorizer, y_train_count_vectorizer, y_test_count_vectorizer = train_test_split(train_vectors1, y_train_data, test_size= 0.2, shuffle= True, random_state= 42)
svc = LinearSVC(dual= False, max_iter= 10000, random_state= 1)
svc.fit(X_train_spacy_vectorizer, y_train_spacy_vectorizer)

print(f'Accuracy Score : {svc.score(X_train_spacy_vectorizer, y_train_spacy_vectorizer):.3f}')

y_pred =svc.predict(X_test_spacy_vectorizer)

cm =confusion_matrix(y_test_spacy_vectorizer, y_pred)
sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')
print(classification_report(y_test_spacy_vectorizer, y_pred))

svc = LinearSVC(dual= False, max_iter= 10000, random_state= 1)
svc.fit(X_train_count_vectorizer, y_train_count_vectorizer)

print(f'Accuracy Score : {svc.score(X_train_count_vectorizer, y_train_count_vectorizer):.3f}')

y_pred_count =svc.predict(X_test_count_vectorizer)
print(classification_report(y_test_count_vectorizer, y_pred_count))
cm =confusion_matrix(y_test_count_vectorizer, y_pred_count)
sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Reds')
logreg_basic = LogisticRegression()
logreg_basic.fit(X_train_spacy_vectorizer, y_train_spacy_vectorizer)
y_pred_logistic_spacy = logreg_basic.predict(X_test_spacy_vectorizer)
print(classification_report(y_test_spacy_vectorizer, y_pred_logistic_spacy))
cm =confusion_matrix(y_test_spacy_vectorizer, y_pred_logistic_spacy)
sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Greens')
logreg_basic = LogisticRegression(C=1.0)
logreg_basic.fit(X_train_count_vectorizer, y_train_count_vectorizer)
y_pred_logistic_count = logreg_basic.predict(X_test_count_vectorizer)
print(classification_report(y_test_count_vectorizer, y_pred_logistic_count))
cm =confusion_matrix(y_test_count_vectorizer, y_pred_logistic_count)
sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Purples')
classifier_spacy = AdaBoostClassifier( DecisionTreeClassifier(max_depth=1),
                                 n_estimators=200,
                                learning_rate=1
                               )
classifier_spacy.fit(X_train_spacy_vectorizer,y_train_spacy_vectorizer)
y_pred_adaboost_spacy = classifier_spacy.predict(X_test_spacy_vectorizer)
print(classification_report(y_test_spacy_vectorizer, y_pred_adaboost_spacy))
cm =confusion_matrix(y_test_spacy_vectorizer, y_pred_adaboost_spacy)
sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')
classifier = AdaBoostClassifier( DecisionTreeClassifier(max_depth=1),
                                 n_estimators=200,
                                learning_rate=1.0
                               )
classifier.fit(X_train_count_vectorizer,y_train_count_vectorizer)
y_pred_adaboost_count = classifier.predict(X_test_count_vectorizer)

print(classification_report(y_test_count_vectorizer, y_pred_adaboost_count))
cm =confusion_matrix(y_test_count_vectorizer, y_pred_adaboost_count)
sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

predicted = cross_val_predict(LogisticRegression(), train_vectors, y_train_data, cv=9)
print( metrics.accuracy_score(y_train_data, predicted))




print (classification_report(y_train_data, predicted))
train_vectors1[0].T.shape
model = Sequential()

model.add(Dense(512, input_dim=16047, activation='relu', name='fc1'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', name='fc2'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', name='fc3'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid', name='output'))

sgd = SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd)
print(model.summary())
history=model.fit(train_vectors1.toarray(), y_train_data, batch_size=4, nb_epoch=20, verbose=1)
print(np.around(model.predict_proba(train_vectors1.toarray())))
from sklearn.metrics import roc_curve, auc

fpr1, tpr1, threshold1 = roc_curve(y_test_spacy_vectorizer, y_pred)
roc_auc1 = auc(fpr1, tpr1)


fpr2, tpr2, threshold2 = roc_curve(y_test_count_vectorizer, y_pred_count)
roc_auc2 = auc(fpr2, tpr2)


fpr3, tpr3, threshold3 = roc_curve(y_test_count_vectorizer, y_pred_logistic_count)
roc_auc3 = auc(fpr3, tpr3)


fpr4, tpr4, threshold4 = roc_curve(y_test_spacy_vectorizer, y_pred_logistic_spacy)
roc_auc4 = auc(fpr4, tpr4)


fpr5, tpr5, threshold5 = roc_curve(y_pred_adaboost_spacy, y_test_spacy_vectorizer)
roc_auc5 = auc(fpr5, tpr5)



fpr6, tpr6, threshold6 = roc_curve(y_pred_adaboost_count, y_test_count_vectorizer)
roc_auc6 = auc(fpr6, tpr6)
plt.figure(figsize=(10,10)) 
plt.plot(fpr1, tpr1, color='navy', lw=2, label='ROC CURVE LINEAR SVM SPACY VECTORIZER'% roc_auc1)
plt.plot(fpr2, tpr2, color='green', lw=2, label='ROC CURVE LINEAR SVM COUNT VECTORIZER'% roc_auc2)
plt.plot(fpr3, tpr3, color='yellow', lw=2, label='ROC CURVE LOGISTIC REGRESSION SPACY VECTORIZER'% roc_auc4)

plt.plot(fpr3, tpr3, color='blue', lw=2, label='ROC CURVE LOGISTIC REGRESSION COUNT VECTORIZER'% roc_auc3)
plt.plot(fpr3, tpr3, color='red', lw=2, label='ROC CURVE ADABOOST CLASSIFIER SPACY VECTORIZER'% roc_auc5)
plt.plot(fpr3, tpr3, color='purple', lw=2, label='ROC CURVE ADABOOST CLASSIFIER COUNT VECTORIZER'% roc_auc6)

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--') 
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('Classifiers ROC curves') 
plt.legend(loc = "lower right")
plt.show()
y_pred_count =svc.predict(test_vectors1)
data_submission = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
data_submission

data_submission["target"] = y_pred_adaboost_count
data_submission = data_submission.loc[:,["id",'target']]
data_submission
def classify(output):
    if output > 0.5:
        return 1
    else:
        return 0
    
data_submission["target"] = data_submission["target"].apply(lambda x : classify(x))
data_submission
data_submission.to_csv('submission.csv', index=False)

