# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB,GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.model_selection import cross_val_score,cross_val_predict

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,precision_recall_curve,roc_auc_score,roc_curve

import gensim

import keras

from keras.models import Sequential

from keras.layers import Dense,Embedding,LSTM,Dropout

from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf

from keras.preprocessing import text, sequence

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/fake-and-real-news-dataset/'

fake = pd.read_csv(path+'Fake.csv')

true = pd.read_csv(path+'True.csv')
fake.head(2)
print('Shape ->',fake.shape)

print('Description ->',fake.describe())

print('Checking null values .. . ',fake.isnull().sum())
plt.figure(figsize=(12,6))

sns.countplot(x='subject',data=fake)

plt.show()
true.head(2)
true.title.value_counts()
print('Shape ->',true.shape)

print('Description ->',true.describe())

print('Checking null values .. . ',true.isnull().sum())
plt.figure(figsize=(12,6))

sns.countplot(x='subject',data=true)

plt.show()
true['category'] = 1

fake['category'] = 0

df = pd.concat([true,fake])

print('Shape ->',df.shape)
sns.countplot(df.category)

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(df.subject)

plt.show()
df['combined_text'] = df['text'] + ' ' + df['title'] + ' ' +df['subject']

del df['title']

del df['text']

del df['subject']

del df['date']
df.head(2)
import re

import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from collections import Counter

nltk.download('stopwords')

nltk.download('wordnet')
STOPWORDS = set(stopwords.words('english'))

punctuations = string.punctuation

STOPWORDS.update(punctuations)
def show_word_cloud(data,title=None):

    word_cloud = WordCloud(

        background_color = 'white',

        max_words =1000,

        width=1600,

        height=800,

        stopwords=STOPWORDS,

        max_font_size = 40, 

        scale = 3,

        random_state = 42 ).generate(data)

    fig = plt.figure(1, figsize = (20, 20))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize = 20)

        fig.subplots_adjust(top = 2.3)



    plt.imshow(word_cloud)

    plt.show()
print(STOPWORDS)
def clean_text(text):

    text = text.lower()

    text = " ".join([word for word in text.split() if word not in STOPWORDS])

    text = re.sub(r'https?://\S+|www\.\S+',r'',text)

    text = re.sub('[\d]',r'',text)

    text = re.sub('[()]',r'',text)

    text = re.sub(r'(<.*?>)',r'',text)

    text = re.sub(r'[^(A-Za-z)]',r' ',text)

    text = re.sub(r'\s+',r' ',text)

  

    return text  
df['text'] = df['combined_text'].apply(lambda x : clean_text(x))
df[:30]
show_word_cloud(" ".join(df[df.category == 1].text),'True Words')
show_word_cloud(" ".join(df[df.category == 0].text),'Fake Words')
fake_len = df[df.category == 0].text.str.len()

true_len = df[df.category == 1].text.str.len()



plt.hist(fake_len, bins=20, label="fake_length")

plt.hist(true_len, bins=20, label="true_length")

plt.legend()

plt.show()
fake_word_token = word_tokenize(" ".join(df[df.category == 0].text))

true_word_token = word_tokenize(" ".join(df[df.category == 1].text))
freq_20_fake = Counter(fake_word_token).most_common(20)

freq_20_true = Counter(true_word_token).most_common(20)
def plot_most_comman_words(data,label):

    most_comman_dict = {}

    palette=''

    for x in data:

        tup = x

        most_comman_dict[tup[0]] = tup[1]

    d = pd.DataFrame({label: list(most_comman_dict.keys()),

                  'Count': list(most_comman_dict.values())})

    if label=='fake words':

        palette='plasma_r'

    else:

        palette='rocket'

    plt.figure(figsize=(12, 8))

    ax = sns.barplot(data=d, x= "Count", y = label,palette=palette)

    ax.set(ylabel = label)

    plt.show()
plot_most_comman_words(freq_20_fake,'fake words')

plot_most_comman_words(freq_20_true,'true words')
X = df.text

y = df.category
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
vect = CountVectorizer()

X_train = vect.fit_transform(x_train)
bow_clf = MultinomialNB()

history = bow_clf.fit(X_train,y_train)
history.score(X_train,y_train)
val_score = cross_val_score(bow_clf,X_train,y_train,cv=10,scoring='accuracy')

print(val_score.mean())
y_train_pred = cross_val_predict(bow_clf,X_train, y_train, cv=10)

y_train_pred
bow_train_cm = confusion_matrix(y_train,y_train_pred)
def cm_scores(cm):

    TN = cm[0][0]

    FP = cm[0][1]

    FN = cm[1][0]

    TP = cm[1][1]

    precision = TP/(TP+FP)

    recall = TP/(TP+FN)

    f1_scre = 2*((precision*recall)/(precision+recall))

    return TN,FP,FN,TP,precision,recall,f1_scre
TN,FP,FN,TP,precision,recall,f1_score = cm_scores(bow_train_cm)

print('Precision : ',precision)

print('Recall : ',recall)

print('F1 Score:',f1_score)
precisions,recalls,thresholds = precision_recall_curve(y_train,y_train_pred)
def plot_precision_vs_recall(precisions,recalls):

    plt.plot(recalls,precisions,'g-',linewidth=2)

    plt.grid(True)

    plt.ylabel('Precision')

    plt.xlabel('Recall')



plt.figure(figsize=(12, 10))
def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):

    plt.figure(figsize=(8, 4))

    plt.plot(thresholds,precisions[:-1], "b--", label="Precision", linewidth=2)

    plt.plot(thresholds,recalls[:-1], "g--", label="Recall", linewidth=2)

    plt.legend(loc="center right", fontsize=16) 

    plt.xlabel("Threshold", fontsize=16)        

    plt.grid(True)                              
def plot_roc_curve(FPR,TPR,label):

    plt.plot(FPR,TPR,'b--',linewidth=2,label=label)

    plt.grid(True)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate (FPR)')

    plt.ylabel('True Positive Rate (TPR)')

    plt.legend(loc='best')

    plt.show()
recalls[np.argmax(precisions >= 0.90)]
plot_precision_vs_recall(precisions,recalls)

plt.plot([0.9636, 0.9636], [0., 0.9], "r:")

plt.plot([0.0, 0.9636], [0.9, 0.9], "r:")

plt.plot([0.9636], [0.9], "ro")

plt.show()
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
FPR,TPR,thresholds = roc_curve(y_train,y_train_pred)
plot_roc_curve(FPR,TPR,'MultinomialNB')
roc_auc_score = roc_auc_score(y_train,y_train_pred)

print('roc_auc_score -- >',roc_auc_score)
X_test = vect.transform(x_test)
predictions = history.predict(X_test)

print(predictions[:10])
bow_cm_test = confusion_matrix(y_test,predictions)

TN,FP,FN,TP,precision,recall,f1_score = cm_scores(bow_cm_test)

print('Precision : ',precision)

print('Recall : ',recall)

print('F1 Score:',f1_score)
tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

X_train = tfidf.fit_transform(x_train.values)
tfidf_clf = MultinomialNB()

history = tfidf_clf.fit(X_train,y_train)

print('Model Score: ',history.score(X_train,y_train))
rf_clf = RandomForestClassifier(n_estimators=500, random_state=42).fit(X_train, y_train)

print('Model Score: ',rf_clf.score(X_train,y_train))
val_score = cross_val_score(tfidf_clf,X_train,y_train,cv=5,scoring='accuracy')

print(val_score.mean())
%time

rf_val_score = cross_val_score(rf_clf,X_train,y_train,cv=5,scoring='accuracy')

print(rf_val_score.mean())
y_train_pred = cross_val_predict(tfidf_clf,X_train, y_train, cv=5)

y_train_pred
rf_y_train_pred = cross_val_predict(rf_clf,X_train, y_train, cv=5)

rf_y_train_pred
tfidf_train_cm = confusion_matrix(y_train,y_train_pred)

rf_tfidf_train_cm = confusion_matrix(y_train,rf_y_train_pred)
TN,FP,FN,TP,precision,recall,f1_score = cm_scores(tfidf_train_cm)

print('Precision : ',precision)

print('Recall : ',recall)

print('F1 Score:',f1_score)
rf_TN,rf_FP,rf_FN,rf_TP,rf_precision,rf_recall,rf_f1_score = cm_scores(rf_tfidf_train_cm)

print('Precision : ',rf_precision)

print('Recall : ',rf_recall)

print('F1 Score:',rf_f1_score)
precisions,recalls,thresholds = precision_recall_curve(y_train,y_train_pred)

rf_precisions,rf_recalls,rf_thresholds = precision_recall_curve(y_train,rf_y_train_pred)
recalls[np.argmax(precisions >= 0.90)]
plot_precision_vs_recall(precisions,recalls)

plt.plot([0.9336, 0.9336], [0., 0.9], "r:")

plt.plot([0.0, 0.9336], [0.9, 0.9], "r:")

plt.plot([0.9336], [0.9], "ro")

plt.show()
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
FPR,TPR,thresholds = roc_curve(y_train,y_train_pred)

rf_FPR,rf_TPR,rf_thresholds = roc_curve(y_train,rf_y_train_pred)
plt.figure(figsize=(8, 6))

plt.plot(rf_FPR,rf_TPR,"g:", linewidth=2,label='Random Forest')

plt.grid(True)

plt.legend(loc='best')

plot_roc_curve(FPR,TPR,'MultinomialNB')
y_train_pred[:4]
X_test = tfidf.transform(x_test)
predictions = history.predict(X_test)

rf_predictions =rf_clf.predict(X_test)

print(rf_predictions[:10])
tfidf_cm_test = confusion_matrix(y_test,predictions)

TN,FP,FN,TP,precision,recall,f1_score = cm_scores(tfidf_cm_test)

print('Precision : ',precision)

print('Recall : ',recall)

print('F1 Score:',f1_score)
rf_tfidf_cm_test = confusion_matrix(y_test,rf_predictions)

rf_TN,rf_FP,rf_FN,rf_TP,rf_precision,rf_recall,rf_f1_score = cm_scores(rf_tfidf_cm_test)

print('Precision : ',rf_precision)

print('Recall : ',rf_recall)

print('F1 Score:',rf_f1_score)
max_features=1000

maxlen = 300
GLOVE_MODEL = '../input/glove-twitter/glove.twitter.27B.100d.txt'
def get_coefs(word, *arr): 

    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(GLOVE_MODEL))
print(embeddings_index.get("leaders"))
tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(x_train)

tokenized_train = tokenizer.texts_to_sequences(x_train)

X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
tokenized_test = tokenizer.texts_to_sequences(x_test)

X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

#change below line if computing normal stats is too slow

embedding_matrix = embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
type(embedding_matrix)
batch_size = 256

epochs = 5

embed_size = 100
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
#Defining Neural Network

model = Sequential()

#Non-trainable embeddidng layer

model.add(Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False))

#LSTM 

model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))

model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))

model.add(Dense(units = 32 , activation = 'relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train, batch_size = batch_size , validation_data = (X_test,y_test) , epochs = epochs , callbacks = [learning_rate_reduction])
y_pred = model.predict_classes(X_test)

y_pred
epochs = [i for i in range(5)]

fig , ax = plt.subplots(1,2)

train_acc = history.history['accuracy']

train_loss = history.history['loss']

val_acc = history.history['val_accuracy']

val_loss = history.history['val_loss']

fig.set_size_inches(20,10)



ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')

ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')

ax[0].set_title('Training & Testing Accuracy')

ax[0].legend()

ax[0].set_xlabel("Epochs")

ax[0].set_ylabel("Accuracy")



ax[1].plot(epochs , train_loss , 'go-' , label = 'Training Loss')

ax[1].plot(epochs , val_loss , 'ro-' , label = 'Testing Loss')

ax[1].set_title('Training & Testing Loss')

ax[1].legend()

ax[1].set_xlabel("Epochs")

ax[1].set_ylabel("Loss")

plt.show()
cm= confusion_matrix(y_test,y_pred)

plt.figure(figsize = (10,10))

sns.heatmap(cm,cmap= "Reds", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Fake','Not Fake'] , yticklabels = ['Fake','Not Fake'])

plt.xlabel("Actual")

plt.ylabel("Predicted")
TN,FP,FN,TP,precision,recall,f1_score = cm_scores(cm)

print('Precision : ',precision)

print('Recall : ',recall)

print('F1 Score:',f1_score)
precisions,recalls,thresholds = precision_recall_curve(y_test,y_pred)
thrs = recalls[np.argmax(precisions >= 0.90)]
plot_precision_vs_recall(precisions,recalls)

plt.plot([thrs, thrs], [0., 1.0], "r:")

plt.plot([0.0, thrs], [1.0, 1.0], "r:")

plt.plot([thrs], [1.0], "ro")

plt.show()
FPR,TPR,thresholds = roc_curve(y_test,y_pred)
plot_roc_curve(FPR,TPR,'LSTM with Glove Emb')