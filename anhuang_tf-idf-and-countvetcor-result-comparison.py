import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

doc = pd.read_csv("../input/spam.csv",encoding='latin-1')
doc = doc.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis = 1)
doc.head()
from sklearn.preprocessing import LabelEncoder

Labels_list = doc['v1'].unique()
Labels = LabelEncoder()
Labels.fit(Labels_list)
doc['v3']=Labels.transform(doc['v1'])
doc.head()
ham_doc = doc.query('v3 == 0')
spam_doc = doc.query('v3 == 1')
print('Number of ham SMS:%d'%len(ham_doc))
print('Number of spam SMS:%d'%len(spam_doc))
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

vectors = TfidfVectorizer(stop_words = 'english')
vectors2 = CountVectorizer(stop_words = 'english')
tfidf = vectors.fit_transform(doc['v2'])
tf = vectors2.fit_transform(doc['v2'])
x_train, x_test, y_train, y_test = train_test_split(tfidf,doc['v3'],test_size = 0.2)
x_train1, x_test1, y_train1, y_test1 = train_test_split(tf,doc['v3'],test_size = 0.2)

model_ = ['NB','RF','LR','SVC','KN']
tfidf_result = []
tf_result = []
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

model1 = MultinomialNB()
model1.fit(x_train, y_train)
tfidf_result.append(accuracy_score(y_test, model1.predict(x_test)))
print('tfidf:%f'% accuracy_score(y_test, model1.predict(x_test)))

model1.fit(x_train1, y_train1)
print('tf:%f' %accuracy_score(y_test1, model1.predict(x_test1)))
tf_result.append(accuracy_score(y_test1, model1.predict(x_test1)))
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
model2.fit(x_train,y_train)
tfidf_result.append(accuracy_score(y_test, model2.predict(x_test)))
print('tfidf:%f'% accuracy_score(y_test, model2.predict(x_test)))

model2.fit(x_train1, y_train1)
print('tf:%f' %accuracy_score(y_test1, model2.predict(x_test1)))
tf_result.append(accuracy_score(y_test1, model2.predict(x_test1)))
from sklearn.svm import SVC

model3 = SVC()
model3.fit(x_train,y_train)
tfidf_result.append(accuracy_score(y_test, model3.predict(x_test)))
print('tfidf:%f'% accuracy_score(y_test, model3.predict(x_test)))

model3.fit(x_train1, y_train1)
print('tf:%f' %accuracy_score(y_test1, model3.predict(x_test1)))
tf_result.append(accuracy_score(y_test1, model3.predict(x_test1)))
from sklearn.linear_model import LogisticRegression

model4 = LogisticRegression()
model4.fit(x_train,y_train)
tfidf_result.append(accuracy_score(y_test, model4.predict(x_test)))
print('tfidf:%f'% accuracy_score(y_test, model4.predict(x_test)))

model4.fit(x_train1, y_train1)
print('tf:%f' %accuracy_score(y_test1, model4.predict(x_test1)))
tf_result.append(accuracy_score(y_test1, model4.predict(x_test1)))
from sklearn.neighbors import KNeighborsClassifier

model5 = KNeighborsClassifier()
model5.fit(x_train,y_train)
tfidf_result.append(accuracy_score(y_test, model5.predict(x_test)))
print('tfidf:%f'% accuracy_score(y_test, model5.predict(x_test)))

model4.fit(x_train1, y_train1)
print('tf:%f' %accuracy_score(y_test1, model5.predict(x_test1)))
tf_result.append(accuracy_score(y_test1, model5.predict(x_test1)))
fig, ax = plt.subplots()
fig.set_size_inches(14, 8)
width = 0.2
ind = np.arange(5)
ax.bar(ind,tfidf_result,width, label = 'tfidf')
ax.bar(ind + width, tf_result, width, label = 'tf')
ax.set_xticklabels(('NB','NB','RF','LR','SVC','KN'))
ax.set_xlabel('algorithms')
ax.set_ylabel('accuracy')
ax.legend()
plt.show()