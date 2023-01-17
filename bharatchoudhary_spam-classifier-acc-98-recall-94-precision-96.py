import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')
data.head()
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True, axis=1)
data.columns = ['label','message']
data.head()
data.describe()
data.groupby(['label']).describe()
sns.countplot(data.label)
data.label.replace({'ham':0,'spam':1}, inplace=True)
plt.figure(figsize=(12,8))
data[data.label==0].message.apply(len).plot(kind='hist',alpha=0.6,bins=35,label='Ham messages')
data[data.label==1].message.apply(len).plot(kind='hist', color='red',alpha=0.6,bins=35,label='Ham messages')

plt.legend()
plt.xlabel("Message Length")
plt.show()
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data[data.label == 0].message))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data[data.label == 1].message))
plt.imshow(wc , interpolation = 'bilinear')
def get_corpus(text):
    words = []
    for i in text:
        for j in i.split():
            words.append(j.strip())
    return words
corpus = get_corpus(data.message)
corpus[:5]
from collections import Counter
counter = Counter(corpus)
most_common = counter.most_common(10)
most_common = dict(most_common)
most_common
sns.barplot(x=list(most_common.values()),y=list(most_common.keys()))
def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
plt.figure(figsize = (16,9))
most_common_bi = get_top_text_ngrams(data.message,10,2)
most_common_bi = dict(most_common_bi)
sns.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()))
plt.figure(figsize = (16,9))
most_common_bi = get_top_text_ngrams(data.message,10,3)
most_common_bi = dict(most_common_bi)
sns.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()))
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
textFeatures = data['message'].copy()
textFeatures = textFeatures.apply(text_process)
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(textFeatures)

X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.3, random_state=111)

clf = LogisticRegressionCV(cv=10, random_state=0)
clf.fit(X_train, y_train)
clf_prediction = clf.predict(X_test)
lg_acc = accuracy_score(y_test,clf_prediction)
print('\n')
print('Accuracy of Naive Bayes =',accuracy_score(y_test,clf_prediction))
print('\n-----------------\n')
print(classification_report(y_test,clf_prediction))
cf_matrix = confusion_matrix(y_true=y_test, y_pred=clf_prediction)
group_names = ['True pos','False Pos','False Neg','True neg']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
plt.figure(figsize=(15,10))

sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label');
mnb = MultinomialNB(alpha=0.2)
mnb.fit(X_train, y_train)
mnb_prediction = mnb.predict(X_test)
mnb_acc=accuracy_score(y_test,mnb_prediction)
print('\n')
print('Accuracy of Naive Bayes =',accuracy_score(y_test,mnb_prediction))
print('\n-----------------\n')
print(classification_report(y_test,mnb_prediction))
cf_matrix = confusion_matrix(y_true=y_test, y_pred=mnb_prediction)
group_names = ['True pos','False Pos','False Neg','True neg']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
plt.figure(figsize=(15,10))

sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label');
svm = SVC(kernel='sigmoid', gamma=1.0)
svm.fit(X_train, y_train)
svm_prediction = svm.predict(X_test)
svm_acc = accuracy_score(y_test,svm_prediction)
print('\n')
print('Accuracy of Naive Bayes =',accuracy_score(y_test,svm_prediction))
print('\n-----------------\n')
print(classification_report(y_test,svm_prediction))
cf_matrix = confusion_matrix(y_true=y_test, y_pred=svm_prediction)
group_names = ['True pos','False Pos','False Neg','True neg']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
plt.figure(figsize=(15,10))

sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label');
rf = RandomForestClassifier(n_estimators=50, max_depth=None, n_jobs=-1)
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
rf_acc = accuracy_score(y_test,rf_prediction)
print('\n')
print('Accuracy of Naive Bayes =',accuracy_score(y_test,rf_prediction))
print('\n-----------------\n')
print(classification_report(y_test,rf_prediction))
cf_matrix = confusion_matrix(y_true=y_test, y_pred=rf_prediction)
group_names = ['True pos','False Pos','False Neg','True neg']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
plt.figure(figsize=(15,10))

sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label');
acc = np.argmax([lg_acc,mnb_acc,svm_acc,rf_acc])
classification = {0:'Logistic Regression',1:'Naive bayes',2:'Support Vector Classifcation',3:'Random Forest Classifier'}
print('Best classifier based on accuracy is {}'.format(classification[acc]))
print('Logistic Regression  is ',precision_score(y_test, clf_prediction))
print('\n')
print('Naive Bayes is ',precision_score(y_test, mnb_prediction))
print('\n')
print('Support Vector Machine Regression is ',precision_score(y_test, svm_prediction))
print('\n')
print('Random Forest Classifer is ',precision_score(y_test, rf_prediction))
print('Logistic Regression  is ',recall_score(y_test, clf_prediction))
print('\n')
print('Naive Bayes is ',recall_score(y_test, mnb_prediction))
print('\n')
print('Support Vector Machine Regression is ',recall_score(y_test, svm_prediction))
print('\n')
print('Random Forest Classifer is ',recall_score(y_test, rf_prediction))