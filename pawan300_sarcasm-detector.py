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
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sns.set_style("whitegrid")
data = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)
data.shape
data = data.drop(["article_link"], axis=1)
data.head()
data.isnull().sum()
sns.countplot(data["is_sarcastic"])
import nltk, re, string
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from wordcloud import WordCloud, STOPWORDS
def text_cleaning(data):
    data = data.apply(lambda x: x.strip().lower())
    data = data.apply(lambda x: re.sub(r'\d+', '', x))
    data = data.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    
    data = data.apply(lambda x : word_tokenize(x))
    data = data.apply(lambda x: [word for word in x if word not in stop_words])
    
    
    lemmatizer = WordNetLemmatizer()
    data = data.apply(lambda x: [lemmatizer.lemmatize(word, pos ='v') for word in x])

    return data
data["headline"] = text_cleaning(data["headline"])
data = data[data["headline"].apply(lambda x: len(x)>0)]
wordcloud = WordCloud(stopwords=STOPWORDS,
                      max_words=2000
                         ).generate(" ".join(list(map(lambda x: " ".join(x), data[data.is_sarcastic==1]["headline"]))))

plt.figure(figsize=(12,12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
wordcloud = WordCloud(stopwords=STOPWORDS,
                      max_words=2000
                         ).generate(" ".join(list(map(lambda x: " ".join(x), data[data.is_sarcastic==0]["headline"]))))

plt.figure(figsize=(12,12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
data["words"] = data["headline"].apply(lambda x: len(x))
data["characters"] = data["headline"].apply(lambda x: len("".join(x)))
plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
plt.xlim(0, 20)
sns.distplot(data[data.is_sarcastic==1]["words"], kde=False)
plt.title("Words distribution in Sarcastic headlines")
plt.grid(False)

plt.subplot(1,2,2)
plt.xlim(0, 15)
sns.distplot(data[data.is_sarcastic==0]["words"], kde=False)
plt.title("Words distribution in Non Sarcastic headlines")

plt.grid(False)
plt.show()
plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
plt.xlim(0, 100)
sns.distplot(data[data.is_sarcastic==1]["characters"], kde=False)
plt.title("Character distribution in Sarcastic headlines")
plt.grid(False)

plt.subplot(1,2,2)
plt.xlim(0, 100)
sns.distplot(data[data.is_sarcastic==0]["characters"], kde=False)
plt.title("Character distribution in Non Sarcastic headlines")

plt.grid(False)
plt.show()
vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform(list(map(lambda x: " ".join(x), data["headline"])))
vector = vector.todense()
print(vector.shape)
xtrain, xtest, ytrain, ytest = train_test_split(vector, data["is_sarcastic"], train_size=0.75)
xtrain.shape, xtest.shape
model = LinearSVC(loss="hinge",fit_intercept=False, max_iter=1500)
model = model.fit(xtrain, ytrain) 
predictions = model.predict(xtest)

svc_train_acc = accuracy_score(ytrain, model.predict(xtrain))
svc_test_acc = accuracy_score(ytest, predictions)
svc_f1_score = f1_score(predictions, ytest)
print("Accuracy score: \n a) Train : {} \n b) Test : {}".format(svc_train_acc, svc_test_acc))
print("Precision score: ", precision_score(ytest, predictions))
print("Recall score: ", recall_score(ytest, predictions))
print("F1 score : ", svc_f1_score)

confusion_matrix(ytest, predictions)
model_lr = LogisticRegression(penalty='l2')
model_lr = model_lr.fit(xtrain, ytrain) 
predictions = model_lr.predict(xtest)

lr_train_acc = accuracy_score(ytrain, model_lr.predict(xtrain))
lr_test_acc = accuracy_score(ytest, predictions)
lr_f1_score = f1_score(predictions, ytest)
print("Accuracy score: \n a) Train : {} \n b) Test : {} ".format(lr_train_acc, lr_test_acc))
print("Precision score: ", precision_score(ytest, predictions))
print("Recall score: ", recall_score(ytest, predictions))
print("F1 score : ", lr_f1_score)

confusion_matrix(ytest, predictions)
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators = 200)
model_rf = model_rf.fit(xtrain, ytrain) 
predictions = model_rf.predict(xtest)

rf_train_acc = accuracy_score(ytrain, model_rf.predict(xtrain))
rf_test_acc = accuracy_score(ytest, predictions)
rf_f1_score = f1_score(predictions, ytest)
print("Accuracy score: \n a) Train : {} \n b) Test : {}".format(rf_train_acc, rf_test_acc))
print("Precision score: ", precision_score(ytest, predictions))
print("Recall score: ", recall_score(ytest, predictions))
print("F1 score : ", rf_f1_score)

confusion_matrix(ytest, predictions)
tree = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)

model_ada = AdaBoostClassifier(base_estimator=tree)
model_ada = model_ada.fit(xtrain, ytrain)
predictions = model_ada.predict(xtest)

ada_train_acc = accuracy_score(ytrain, model_ada.predict(xtrain))
ada_test_acc = accuracy_score(ytest, predictions)
ada_f1_score = f1_score(predictions, ytest)
print("Accuracy score: \n a) Train : {}\n b) Test : {}".format(ada_train_acc, ada_test_acc))
print("Precision score: ", precision_score(ytest, predictions))
print("Recall score: ", recall_score(ytest, predictions))
print("F1 score : ",  ada_f1_score)

confusion_matrix(ytest, predictions)
model = ExtraTreesClassifier(bootstrap=False, criterion='gini', max_depth= None, 
                             max_features= 3, min_samples_leaf= 1, min_samples_split= 10, 
                             n_estimators= 300)
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

etc_train_acc = accuracy_score(ytrain, model.predict(xtrain))
etc_test_acc = accuracy_score(ytest, predictions)
etc_f1_score = f1_score(predictions, ytest)
print("Accuracy score: \n a) Train : {}\n b) Test : {}".format(etc_train_acc, etc_test_acc))
print("Precision score: ", precision_score(ytest, predictions))
print("Recall score: ", recall_score(ytest, predictions))
print("F1 score : ", etc_f1_score)

confusion_matrix(ytest, predictions)
xtrain = np.array(xtrain)
xtest = np.array(xtest)
xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1],-1)
xtest = xtest.reshape(xtest.shape[0], xtest.shape[1],-1)
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(xtrain[0].shape)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
history = model.fit(xtrain, ytrain, epochs=1)
train_acc = []
test_acc = []
f1 = []

for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    print("For i =",i)
    predictions = model.predict(xtest)
    predictions = [1 if j>i else 0 for j in predictions]
    train_predict = model.predict(xtrain)
    train_predict = [1 if j>i else 0 for j in train_predict]
    rnn_train_acc = accuracy_score(ytrain, train_predict)
    rnn_test_acc = accuracy_score(ytest, predictions)
    rnn_f1_score = f1_score(predictions, ytest)
    train_acc.append(rnn_train_acc)
    test_acc.append(rnn_test_acc)
    f1.append(rnn_f1_score)
    print("Accuracy score: \n a) Train : {}\n b) Test : {}".format(rnn_train_acc, rnn_test_acc))
    print("Precision score: ", precision_score(ytest, predictions))
    print("Recall score: ", recall_score(ytest, predictions))
    print("F1 score : ", rnn_f1_score)

    confusion_matrix(ytest, predictions)
temp = np.argmax(rnn_f1_score)
rnn_train_acc = train_acc[temp]
rnn_test_acc =  test_acc[temp]
rnn_f1_score =  f1[temp]
models = [('Linear SVC', svc_test_acc, svc_test_acc, svc_f1_score),
          ('Logistic Regression', lr_train_acc, lr_test_acc, lr_f1_score),
          ('Random Forest', rf_train_acc, rf_test_acc, rf_f1_score),
          ('Ada boost', ada_train_acc, ada_test_acc, ada_f1_score),
          ('Extra tree', etc_train_acc, etc_test_acc, etc_f1_score),
          ('RNN', rnn_train_acc, rnn_test_acc, rnn_f1_score)
         ]
         
predict = pd.DataFrame(data = models, columns=['Model', 'Train accuracy', 'Test accuracy', 'F1 score'])
predict
f, axe = plt.subplots(1,1, figsize=(12,6))

predict.sort_values(by=['F1 score'], ascending=False, inplace=True)

sns.barplot(x='F1 score', y='Model', data = predict, ax = axe, palette='inferno')
axe.set_xlabel('F1 score', size=16)
axe.set_ylabel('Model')
axe.set_xlim(0,1.0)
axe.set_xticks(np.arange(0, 1.1, 0.1))
plt.show()
import time, itertools
from gensim.models import Word2Vec
w2v_model = Word2Vec(window=2,
                     min_count=0,
                     size=500,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20)
t = time.time()

w2v_model.build_vocab(list(data["headline"]), progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))
t = time.time()

w2v_model.train(list(data["headline"]), total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time.time() - t) / 60, 2)))
w2v_model.init_sims(replace=True)
data_vector = []
for i in range(len(data)):
     data_vector.append(np.mean(w2v_model[data["headline"].iloc[i]], axis=0))
xtrain, xtest, ytrain, ytest = train_test_split(np.array(data_vector), data["is_sarcastic"], train_size = 0.75)
model = LinearSVC(loss="hinge",fit_intercept=False, max_iter=1500)
model = model.fit(xtrain, ytrain) 
predictions = model.predict(xtest)

svc_train_acc = accuracy_score(ytrain, model.predict(xtrain))
svc_test_acc = accuracy_score(ytest, predictions)
svc_f1_score = f1_score(predictions, ytest)
print("Accuracy score: \n a) Train : {} \n b) Test : {}".format(svc_train_acc, svc_test_acc))
print("Precision score: ", precision_score(ytest, predictions))
print("Recall score: ", recall_score(ytest, predictions))
print("F1 score : ", svc_f1_score)

confusion_matrix(ytest, predictions)
model_lr = LogisticRegression(penalty='l2')
model_lr = model_lr.fit(xtrain, ytrain) 
predictions = model_lr.predict(xtest)

lr_train_acc = accuracy_score(ytrain, model_lr.predict(xtrain))
lr_test_acc = accuracy_score(ytest, predictions)
lr_f1_score = f1_score(predictions, ytest)
print("Accuracy score: \n a) Train : {} \n b) Test : {} ".format(lr_train_acc, lr_test_acc))
print("Precision score: ", precision_score(ytest, predictions))
print("Recall score: ", recall_score(ytest, predictions))
print("F1 score : ", lr_f1_score)

confusion_matrix(ytest, predictions)
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators = 200)
model_rf = model_rf.fit(xtrain, ytrain) 
predictions = model_rf.predict(xtest)

rf_train_acc = accuracy_score(ytrain, model_rf.predict(xtrain))
rf_test_acc = accuracy_score(ytest, predictions)
rf_f1_score = f1_score(predictions, ytest)
print("Accuracy score: \n a) Train : {} \n b) Test : {}".format(rf_train_acc, rf_test_acc))
print("Precision score: ", precision_score(ytest, predictions))
print("Recall score: ", recall_score(ytest, predictions))
print("F1 score : ", rf_f1_score)

confusion_matrix(ytest, predictions)
tree = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)

model_ada = AdaBoostClassifier(base_estimator=tree)
model_ada = model_ada.fit(xtrain, ytrain)
predictions = model_ada.predict(xtest)

ada_train_acc = accuracy_score(ytrain, model_ada.predict(xtrain))
ada_test_acc = accuracy_score(ytest, predictions)
ada_f1_score = f1_score(predictions, ytest)
print("Accuracy score: \n a) Train : {}\n b) Test : {}".format(ada_train_acc, ada_test_acc))
print("Precision score: ", precision_score(ytest, predictions))
print("Recall score: ", recall_score(ytest, predictions))
print("F1 score : ",  ada_f1_score)

confusion_matrix(ytest, predictions)
model = ExtraTreesClassifier(bootstrap=False, criterion='gini', max_depth= None, 
                             max_features= 3, min_samples_leaf= 1, min_samples_split= 10, 
                             n_estimators= 300)
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

etc_train_acc = accuracy_score(ytrain, model.predict(xtrain))
etc_test_acc = accuracy_score(ytest, predictions)
etc_f1_score = f1_score(predictions, ytest)
print("Accuracy score: \n a) Train : {}\n b) Test : {}".format(etc_train_acc, etc_test_acc))
print("Precision score: ", precision_score(ytest, predictions))
print("Recall score: ", recall_score(ytest, predictions))
print("F1 score : ", etc_f1_score)

confusion_matrix(ytest, predictions)
xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1]
               ,-1)
xtest = xtest.reshape(xtest.shape[0], xtest.shape[1]
               ,-1)
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(xtrain[0].shape)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, "sigmoid")
])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
history = model.fit(xtrain, ytrain, epochs=50)
train_acc = []
test_acc = []
f1 = []

for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    print("For i =",i)
    predictions = model.predict(xtest)
    predictions = [1 if j>i else 0 for j in predictions]
    train_predict = model.predict(xtrain)
    train_predict = [1 if j>i else 0 for j in train_predict]
    rnn_train_acc = accuracy_score(ytrain, train_predict)
    rnn_test_acc = accuracy_score(ytest, predictions)
    rnn_f1_score = f1_score(predictions, ytest)
    train_acc.append(rnn_train_acc)
    test_acc.append(rnn_test_acc)
    f1.append(rnn_f1_score)
    print("Accuracy score: \n a) Train : {}\n b) Test : {}".format(rnn_train_acc, rnn_test_acc))
    print("Precision score: ", precision_score(ytest, predictions))
    print("Recall score: ", recall_score(ytest, predictions))
    print("F1 score : ", rnn_f1_score)

    confusion_matrix(ytest, predictions)
temp = np.argmax(rnn_f1_score)
rnn_train_acc = train_acc[temp]
rnn_test_acc =  test_acc[temp]
rnn_f1_score =  f1[temp]
models = [('Linear SVC', svc_test_acc, svc_test_acc, svc_f1_score),
          ('Logistic Regression', lr_train_acc, lr_test_acc, lr_f1_score),
          ('Random Forest', rf_train_acc, rf_test_acc, rf_f1_score),
          ('Ada boost', ada_train_acc, ada_test_acc, ada_f1_score),
          ('Extra tree', etc_train_acc, etc_test_acc, etc_f1_score),
          ('RNN', rnn_train_acc, rnn_test_acc, rnn_f1_score)
         ]
         
predict = pd.DataFrame(data = models, columns=['Model', 'Train accuracy', 'Test accuracy', 'F1 score'])
predict
f, axe = plt.subplots(1,1, figsize=(12,6))

predict.sort_values(by=['F1 score'], ascending=False, inplace=True)

sns.barplot(x='F1 score', y='Model', data = predict, ax = axe, palette='inferno')
axe.set_xlabel('F1 score', size=16)
axe.set_ylabel('Model')
axe.set_xlim(0,1.0)
axe.set_xticks(np.arange(0, 1.1, 0.1))
plt.show()
