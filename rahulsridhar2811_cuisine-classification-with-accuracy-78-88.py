import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
file = r'../input/recipe-ingredients-dataset/train.json'
with open(file) as train_file:
    dict_train = json.load(train_file)

print(dict_train[0])
len(dict_train)
dict_train[0]['ingredients']
id_ = []
cuisine = []
ingredients = []
for i in range(len(dict_train)):
    id_.append(dict_train[i]['id'])
    cuisine.append(dict_train[i]['cuisine'])
    ingredients.append(dict_train[i]['ingredients'])
    
    
    
import pandas as pd
df = pd.DataFrame({'id':id_, 
                   'cuisine':cuisine, 
                   'ingredients':ingredients})
print(df.head(5))
df['cuisine'].value_counts()
new = []
for s in df['ingredients']:
    s = ' '.join(s)
    new.append(s)
    
df['ing'] = new
import re
l=[]
for s in df['ing']:
    
    #Remove punctuations
    s=re.sub(r'[^\w\s]','',s)
    
    #Remove Digits
    s=re.sub(r"(\d)", "", s)
    
    #Remove content inside paranthesis
    s=re.sub(r'\([^)]*\)', '', s)
    
    #Remove Brand Name
    s=re.sub(u'\w*\u2122', '', s)
    
    #Convert to lowercase
    s=s.lower()
    
    #Remove Stop Words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(s)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    s=' '.join(filtered_sentence)
    
    #Remove low-content adjectives
    
    
    #Porter Stemmer Algorithm
    words = word_tokenize(s)
    word_ps=[]
    for w in words:
       word_ps.append(ps.stem(w))
    s=' '.join(word_ps)
    
    l.append(s)
df['ing_mod']=l
print(df.head(10))
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['ing_mod'])

print(X)
#print(vectorizer.get_feature_names())
len(new)
type(df['ing'][0])
s='1 1cool co1l coo1'
s=re.sub(r"(\d)", "", s)
print(s)
s='hi 1(bye)'
s=re.sub(r'\([^)]*\)', '', s)
print(s)
s='hi 1 Marvel™ hi'
s=re.sub(u'\w*\u2122', '', s)
print(s)
import re
from nltk.corpus import stopwords
s="I love this phone, its super fast and there's so much new and cool things with jelly bean....but of recently I've seen some bugs."
pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
s = pattern.sub('', s)
print(s)
#"love phone, super fast much cool jelly bean....but recently bugs."
import nltk
nltk.download('word_tokenize')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)
"love phone, super fast much cool jelly bean....but recently bugs."
import nltk
nltk.download('word_tokenize')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

s = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."

print(s)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['cuisine'])
df['cuisine']=le.transform(df['cuisine']) 
df['cuisine'].value_counts()
cuisine_map={'0':'brazilian', '1':'british', '2':'cajun_creole', '3':'chinese', '4':'filipino', '5':'french', '6':'greek', '7':'indian', '8':'irish', '9':'italian', '10':'jamaican', '11':'japanese', '12':'korean', '13':'mexican', '14':'moroccan', '15':'russian', '16':'southern_us', '17':'spanish', '18':'thai', '19':'vietnamese'}
Y=[]
Y = df['cuisine']

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)
for K in range(25):
 K_value = K+1
 neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
 neigh.fit(X_train, y_train) 
 y_pred = neigh.predict(X_test)
 print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)

#Implement KNN(So we take K value to be 11)
neigh = KNeighborsClassifier(n_neighbors = 11, weights='uniform', algorithm='auto')
neigh.fit(X_train, y_train) 
y_pred = neigh.predict(X_test)
print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:11")
#Implement Grid Serch for best Gamma, C and Selection between rbf and linear kernel
from sklearn import svm, datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
clf.fit(X_train, y_train)   
print('Best score for data1:', clf.best_score_) 
print('Best C:',clf.best_estimator_.C) 
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)
#OVA SVM(Grid Search Results: Kernel - Linear, C - 1, Gamma - Auto)
from sklearn import svm
lin_clf = svm.LinearSVC(C=1)
lin_clf.fit(X_train, y_train)
y_pred=lin_clf.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)
#SVM by Crammer(Grid Search Results: Gamma - , C - )
lin_clf = svm.LinearSVC(C=1.0, multi_class='crammer_singer')
lin_clf.fit(X_train, y_train)
y_pred=lin_clf.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)
#Implementing OVA Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)
#Implementing OVA Logistic Regerssion
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)
df3=pd.DataFrame({'id':y_test.index, 'cuisine':y_test.values})
y_test2=[]
y_test1=df3['cuisine'].tolist()
for i in range(len(df3['cuisine'])):
    y_test2.append(cuisine_map[str(df3['cuisine'][i])])
print(y_test2)
#Convert Predicted Output 
#_______ gives the best accuracy. So lets implement it on last time to get the final output.
y_pred1=[]
for i in range(len(y_pred)):
    y_pred1.append(cuisine_map[str(y_pred[i])])
print(y_test2)
#We Choose OVA SVM as it gives the best accuracy of 78.88%
from sklearn import svm
lin_clf = svm.LinearSVC(C=1)
lin_clf.fit(X_train, y_train)
y_pred=lin_clf.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)
result=pd.DataFrame({'Actual Cuisine':y_test2, 'Predicted Cuisine':y_test2})
print(result)

