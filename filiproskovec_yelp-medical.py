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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import re, string
import sys
import time
from nltk.corpus import stopwords
from wordcloud import WordCloud
from mpl_toolkits.basemap import Basemap

from subprocess import check_output
%matplotlib inline
import json


def read_json(file):
    def init_ds(json):
        ds= {}
        keys = json.keys()
        for k in keys:
            ds[k]= []
        return ds, keys

    dataset = {}
    keys = []
    with open(file) as file_lines:
        for count, line in enumerate(file_lines):
            data = json.loads(line.strip())
            if count ==0:
                dataset, keys = init_ds(data)
            for k in keys:
                dataset[k].append(data[k])
                
        return pd.DataFrame(dataset)
%%time

yelp_business= read_json('../input/yelp-dataset/yelp_academic_dataset_business.json')
yelp_business.head()
# distribution of ratings
x=yelp_business['stars'].value_counts()
x=x.sort_index()
#plot
fig = plt.figure(figsize=(10,5))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Distribution of Star Ratings")
plt.ylabel('number of businesses', fontsize=12)
plt.xlabel('Star Ratings ', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
#ax.set_facecolor('#269cbc')
#fig.patch.set_facecolor('#269cbc')
fig.savefig('stars_general.png', bbox_inches='tight')
plt.show()

#popular business categories

business_cats=' '.join(yelp_business['categories'].dropna())

cats=pd.DataFrame(business_cats.split(','),columns=['category'])
x=cats.category.value_counts()
print("There are ",len(x)," different types/categories of Businesses in Yelp!")
#prep for chart
x=x.sort_values(ascending=False)
x=x.iloc[0:20]

fig = plt.figure(figsize=(10,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)#,color=color[5])
plt.title("Top categories",fontsize=25)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('Number of businesses', fontsize=12)
plt.xlabel('Category', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
fig.savefig('popular_categories2.png', bbox_inches='tight')
%%time

#yelp_business= read_json('../input/yelp-dataset/yelp_academic_dataset_business.json')
#restaurants = yelp_business[yelp_business['categories'].str.contains('Restaurant') == True]
medical = yelp_business[yelp_business['categories'].str.contains('Medical') == True]
del(yelp_business)
medical.describe()
print('Median and average review count',medical['review_count'].median(), medical['review_count'].mean())
#print('Median over 1000: ', medical[medical['review_count'] >1000]['review_count'].median())
plt.figure(figsize = (10,10))
sns.barplot(medical[medical['review_count'] >320]['review_count'],medical[medical['review_count'] >320]['name'],
           palette = 'summer')
plt.xlabel('')
plt.title('Top review count');
plt.savefig("top_reviewed_places.png", bbox_inches='tight')
# distribution of ratings in medical and healt care
x=medical['stars'].value_counts()
x=x.sort_index()
#plot
plt.figure(figsize=(10,5))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Average Star Ratings in Medical and Health care")
plt.ylabel('number of businesses', fontsize=12)
plt.xlabel('star ratings ', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
#ax.set_facecolor((1.0, 0.47, 0.42))
plt.show()
%%time

yelp_review= read_json('../input/yelp-dataset/yelp_academic_dataset_review.json')

#review_restaurants = yelp_review[yelp_review.business_id.isin(restaurants['business_id']) == True]

reviews = yelp_review[yelp_review.business_id.isin(medical['business_id']) == True]
del(yelp_review)
# reviews =review_restaurants.drop(['review_id','user_id','business_id','date','useful','funny','cool'],axis=1)
reviews = reviews.drop(['review_id','user_id','business_id','date','useful','funny','cool'],axis=1)
reviews.head(10)
reviews.describe()

reviews.count()
reviews['stars'].value_counts()

cloud_texts = ' '.join(reviews['text'][:10].astype(str))
stopwords = {'one','some','is','a','at','is','he','back', 'if', 'the'}
cloud_texts  = ' '.join([word for word in re.split("\W+",cloud_texts) if word.lower() not in stopwords])
cloud = WordCloud(width=1440, height= 1080,max_words= 200, background_color='white').generate(cloud_texts)
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off');
plt.savefig("cloud_text.png", bbox_inches='tight')
# distribution of ratings in reviews on medical and healt care
x=reviews['stars'].value_counts()
x=x.sort_index()
#plot
plt.figure(figsize=(10,5))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Distribution of Star Ratings in Medical and Health care reviews")
plt.ylabel('number of reviews', fontsize=12)
plt.xlabel('star ratings ', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()

#sns.countplot(reviews.stars)
reviews.count()
# reviews = reviews[reviews.stars!=3]
# reviews['label'] = reviews['stars'].apply(lambda x: 1 if x>3 else 0)
# reviews = reviews.drop('stars',axis=1)

# negative/neutral/positive
reviews = reviews[reviews['stars'].isin([1,3,5])]
reviews['label'] = reviews['stars'].apply(lambda x: 0 if x==1 else x)
reviews['label'] = reviews['label'].apply(lambda x: 1 if x==3 else x)
reviews['label'] = reviews['label'].apply(lambda x: 2 if x==5 else x)

print(reviews['stars'].value_counts())
from sklearn.utils import shuffle
reviews2 = reviews.groupby('label')[['text', 'label']].apply(lambda s: s.sample(10000))
reviews2.reset_index(drop=True, inplace=True)
reviews2 = shuffle(reviews2)
#reviews2 = reviews[:100000]
print(reviews2.head())
print(reviews2['label'].value_counts())
neg_reviews = reviews2[reviews2['label'] == 0].sample(1) 
print("Negative reviews:", neg_reviews['text'].values)
neutr_reviews = reviews2[reviews2['label'] == 1].sample(1) 
print("Neutral reviews:", neutr_reviews['text'].values)
pos_reviews = reviews2[reviews2['label'] == 2].sample(1) 
print("Positive reviews:", pos_reviews['text'].values)
import pandas as pd,numpy as np,seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib inline
import spacy
from keras.layers import Bidirectional,GlobalMaxPool1D,Conv1D
from keras.layers import LSTM,Input,Dense,Dropout,Activation
from keras.models import Model
texts = reviews2["text"].values
labels = reviews2["label"].values
texts.shape, labels.shape
max_num_words = 1000
tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
len(word_index)
#print(tokenizer.word_counts)
# number of input texts
# print(tokenizer.document_count)
# print(tokenizer.word_index)
# print(tokenizer.word_docs)
seq_sample = tokenizer.texts_to_sequences(["hello everybody, how are you doing SACSjk"])
print(seq_sample)
lengths = np.array([len(i) for i in sequences])
print("Average lenngth:", np.mean(lengths), "Median of length:", np.median(lengths))
from sklearn.model_selection import train_test_split
# maximal sequence length based on the median of the review word lengths
max_seq_length = 80
data = pad_sequences(sequences, maxlen=max_seq_length, truncating='pre') #'post'
labels = to_categorical(np.asarray(labels))
print(data.shape, labels.shape)
print(len(texts[1]), texts[1])
print(len(data[0]), data[0])
text_sample = tokenizer.sequences_to_texts([data[1]])
print(text_sample)
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
unique_elements, counts_elements = np.unique(np.argmax(Y_train,axis=1), return_counts=True)
print("Frequency of unique values of Y_train:",unique_elements, counts_elements)
def read_glove_vecs(glove_file):
    embedding_index = {}

    with open(glove_file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_index[word] = coefs
        
    return embedding_index
embedding_index = read_glove_vecs('../input/glove6b50dtxt/glove.6B.50d.txt')
print('found word vecs: ',len(embedding_index))
def set_embedding_matrix(word_index, embedding_dim): 
    #embedding_dim = 50
    embedding_matrix = np.zeros((len(word_index)+1,embedding_dim))
    # embedding_matrix.shape 
    for word,i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
embedding_dim = 50
embedding_matrix = set_embedding_matrix(word_index, embedding_dim)
print(max_seq_length)
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index)+1,embedding_dim,weights=[embedding_matrix],input_length=max_seq_length,trainable=False)
from keras.layers import Bidirectional,GlobalMaxPooling1D
from keras.layers import LSTM,Input,Dense,Dropout,Activation
from keras.models import Model
def LSTM_Model(max_seq_length):
    inp = Input(shape=(max_seq_length,))
    x = embedding_layer(inp)
    x = Bidirectional(LSTM(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)    
    x = GlobalMaxPooling1D()(x)    
    x = Dense(64,activation='relu')(x)
    x = Dropout(0.1)(x)
    
    # positive /negative
    #x = Dense(2,activation='sigmoid')(x) 
    # positive /negative /neutral 
    x = Dense(3,activation='sigmoid')(x)  
    x = Activation('softmax')(x)
    
    model = Model(inputs=inp,outputs=x)
    return model

model = LSTM_Model(max_seq_length)
#model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=20,batch_size=128);
import pickle
# save the model to disk
filename_nn = 'nn_model.pkl'
pickle.dump(model, open(filename_nn, 'wb'))
# load the model from disk
#loaded_model = pickle.load(open(filename_nn, 'rb'))

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("learning_curve.png", bbox_inches='tight')
plt.show()
import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
from sklearn.metrics import confusion_matrix
import itertools
pred = np.argmax(model.predict(X_test),axis=1)
true_label = np.argmax(Y_test, axis=1)


cm = confusion_matrix(y_true=true_label, y_pred=pred)
plot_confusion_matrix(cm, target_names=['neg', 'neutr', 'pos'], normalize=True) #,cmap=plt.cm.Blues):
from sklearn.metrics import classification_report
print(classification_report(true_label, pred))
for i in range(100):
    text_sample = tokenizer.sequences_to_texts([data[i]])
    
    #print(np.argmax(labels[i]))
    sample = data[i].reshape(1,80)
    #print(type(data[i]), data[i].shape)
    true_label = np.argmax(labels[i])
    pred_label = np.argmax(model.predict(sample))
    if true_label != pred_label:
        print(text_sample)
        print("True label: ", true_label,"Predicted label: ", pred_label)


my_text_neg = "In one word it was terrible."
my_text_neutral = "Eventhough not very warm the doctor treated me nicely and there were no problems in the end"
my_text_pos = "Absolutely stunning experience! I am looking forward to next visit."
for my_text in [my_text_neg, my_text_neutral, my_text_pos]:
    my_seq = tokenizer.texts_to_sequences([my_text])
    print("Recontructed text: ", tokenizer.sequences_to_texts(my_seq))
    my_seq = pad_sequences(my_seq, maxlen=max_seq_length, truncating='pre') 
    print("Predicted probabilities: ", model.predict(my_seq.reshape(1,80)))
    print("Predicted label: ", np.argmax(model.predict(my_seq.reshape(1,80))))



