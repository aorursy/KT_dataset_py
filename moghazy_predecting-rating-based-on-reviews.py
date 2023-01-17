# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec 
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.naive_bayes import GaussianNB
import re
import spacy
from sklearn.utils import shuffle

from keras.utils.np_utils import to_categorical  
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, confusion_matrix, recall_score
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

import keras

drugsComTrain_raw = pd.read_csv('../input/drugsComTrain_raw.csv')
drugsComTest_raw = pd.read_csv('../input/drugsComTest_raw.csv')

drugsComTrain_raw.shape

# show number of samples per class
drugsComTrain_raw.rating.value_counts()

plt.hist(drugsComTrain_raw.rating)
plt.show()

d = {1: 1, 
     2: 2, 3:2, 
     4: 3, 5: 3,
     6: 4, 7: 4,
     8: 5, 9: 5,
     10: 6}

drugsComTrain_raw['new_ratings'] = drugsComTrain_raw.rating.map(d)
drugsComTest_raw['new_ratings'] = drugsComTest_raw.rating.map(d)

# We will  not remove stop words for this case
'not' in set(stopwords.words('english'))


# # # https://www.kaggle.com/shashanksai/text-preprocessing-using-python
# snow = nltk.stem.SnowballStemmer('english')
all_reviews = []
for item in drugsComTrain_raw.review:
    temp = item
    temp = temp.lower()
    cleanr = re.compile('<.*?>')
    temp = re.sub(cleanr, ' ', temp)
    temp = re.sub(r'[?|!|\'|"|#]',r'',temp)
    temp = re.sub(r'[.|,|)|(|\|/]',r'',temp) 
#     temp = [word for word in temp.split(' ') if word not in set(stopwords.words('english'))]
    all_reviews.append(temp)
    
all_reviews_test = []
for item in drugsComTest_raw.review:
    temp = item
    temp = temp.lower()
    cleanr = re.compile('<.*?>')
    temp = re.sub(cleanr, ' ', temp)
    temp = re.sub(r'[?|!|\'|"|#]',r'',temp)
    temp = re.sub(r'[.|,|)|(|\|/]',r'',temp) 
#     temp = [word for word in temp.split(' ') if word not in set(stopwords.words('english'))]
    all_reviews_test.append(temp)

X = all_reviews
Y = drugsComTrain_raw.new_ratings
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)

x_test = all_reviews_test
y_test = drugsComTest_raw.new_ratings

vectorizer = TfidfVectorizer(ngram_range=(1,2))

x_train_vectors = vectorizer.fit_transform(x_train)
x_valid_vectors = vectorizer.transform(x_valid)
x_test_vectors = vectorizer.transform(x_test)

selector = SelectKBest(f_classif, k=min(1000, x_train_vectors.shape[1]))
selector.fit(x_train_vectors, y_train)
x_train_vectors = selector.transform(x_train_vectors).astype('float32')

x_valid_vectors = selector.transform(x_valid_vectors).astype('float32')
x_test_vectors = selector.transform(x_test_vectors).astype('float32')

x_train_vectors.shape, y_train.shape

round(drugsComTrain_raw.new_ratings.value_counts() * 10 / drugsComTrain_raw.shape[0])

few_classes = list(map(lambda x: x in list(range(2, 6)), y_train.tolist()))

more_classes = list(map(lambda x: x in [1, 6], y_train.tolist()))

np.sum(few_classes)

few_classes = np.array(few_classes)
more_classes = np.array(more_classes)

x_train_few = x_train_vectors[few_classes]
y_train_few = y_train[few_classes]

x_train_more = x_train_vectors[more_classes]
y_train_more = y_train[more_classes]

all_x_train = list(x_train_few)
all_y_train = list(y_train_few)

for i in range(2):
    all_x_train.extend(x_train_few)
    all_y_train.extend(y_train_few)
    
all_x_train.extend(x_train_more)
all_y_train.extend(y_train_more)

all_x_train = np.concatenate([x_train_more.toarray(), x_train_few.toarray(), x_train_few.toarray(), x_train_few.toarray()], axis=0)
all_y_train = np.concatenate([y_train_more, y_train_few, y_train_few, y_train_few], axis=0)

all_y_train.shape

new_x_train, new_y_train = shuffle(all_x_train, all_y_train)

new_x_train.shape




# clf = ExtraTreesClassifier(n_estimators=100, random_state=0, class_weight='balanced')

# clf.fit(x_train_vectors, y_train)

# print('Accuracy of classifier on training set: {:.2f}'.format(clf.score(x_train_vectors, y_train) * 100))
# print('Accuracy of classifier on test set: {:.2f}'.format(clf.score(x_test_vectors, y_test) * 100))
# y_pred = clf.predict(x_test_vectors)
# precision_score(y_test, y_pred, average='macro')

# precision_score(y_test, y_pred, average='micro')

# recall_score(y_test, y_pred, average='macro')

# recall_score(y_test, y_pred, average='micro')

# classification_report(y_test, y_pred2, target_names=["class 1", "class 2", "class 3", "class 4", "class 5", "class 6"])

# cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('ExtraTreesClassifier')
# plt.colorbar()
# tick_marks = np.arange(1, 11)
# plt.xticks(tick_marks, np.arange(1, 11))
# plt.yticks(tick_marks, np.arange(1, 11))

# np.sum(y_pred == y_test) / len(y_test)

new_x_train.shape

# https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
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

model_mlp = keras.models.Sequential()

model_mlp.add(keras.layers.Dense(200, input_shape=(1000,)))
model_mlp.add(keras.layers.BatchNormalization())
model_mlp.add(keras.layers.Activation('relu'))
model_mlp.add(keras.layers.Dropout(0.5))

model_mlp.add(keras.layers.Dense(300))
model_mlp.add(keras.layers.BatchNormalization())
model_mlp.add(keras.layers.Activation('relu'))
model_mlp.add(keras.layers.Dropout(0.5))

model_mlp.add(keras.layers.Dense(100, activation='relu'))
model_mlp.add(keras.layers.Dense(7, activation='sigmoid'))

model_mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_mlp.summary()

y_train_one_hot = to_categorical(new_y_train)

y_valid_one_hot = to_categorical(y_valid)
y_test_one_hot = to_categorical(y_test)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.00001)
history = model_mlp.fit(new_x_train, y_train_one_hot, epochs=25, batch_size=64, validation_data=(x_valid_vectors, y_valid_one_hot), callbacks=[reduce_lr])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

y_pred2 = model_mlp.predict_classes(x_test_vectors)

classification_report(y_test, y_pred2, target_names=["class 1", "class 2", "class 3", "class 4", "class 5", "class 6"])

print(precision_score(y_test, y_pred2, average='macro'))

print(recall_score(y_test, y_pred2, average='macro'))

cm2 = confusion_matrix(y_pred=y_pred2, y_true=y_test)
plot_confusion_matrix(cm           = cm2, 
                      normalize    = False,
                      target_names = ["class 1", "class 2", "class 3", "class 4", "class 5", "class 6"],
                      title        = "Confusion Matrix")

print('Accuracy = {}'.format(np.sum(y_pred2 == y_test) / len(y_test)))

model_mlp2 = keras.models.Sequential()

model_mlp2.add(keras.layers.Dense(200, input_shape=(1000,)))
model_mlp2.add(keras.layers.BatchNormalization())
model_mlp2.add(keras.layers.Activation('relu'))
model_mlp2.add(keras.layers.Dropout(0.2))

model_mlp2.add(keras.layers.Dense(300))
model_mlp2.add(keras.layers.BatchNormalization())
model_mlp2.add(keras.layers.Activation('relu'))
model_mlp2.add(keras.layers.Dropout(0.2))

model_mlp2.add(keras.layers.Dense(100, activation='relu'))
model_mlp2.add(keras.layers.Dense(7, activation='sigmoid'))

model_mlp2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

reduce_lr2 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.00001)
history2 = model_mlp2.fit(new_x_train, y_train_one_hot, epochs=25, batch_size=64, validation_data=(x_valid_vectors, y_valid_one_hot), callbacks=[reduce_lr2])

plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

y_pred2_2 = model_mlp2.predict_classes(x_test_vectors)

classification_report(y_test, y_pred2_2, target_names=["class 1", "class 2", "class 3", "class 4", "class 5", "class 6"])

print(precision_score(y_test, y_pred2_2, average='macro'))

print(recall_score(y_test, y_pred2_2, average='macro'))

cm2_2 = confusion_matrix(y_pred=y_pred2_2, y_true=y_test)

plot_confusion_matrix(cm           = cm2_2, 
                      normalize    = False,
                      target_names = ["class 1", "class 2", "class 3", "class 4", "class 5", "class 6"],
                      title        = "Confusion Matrix")

print('Accuracy = {}'.format(np.sum(y_pred2_2 == y_test) / len(y_test)))


model_mlp3 = keras.models.Sequential()

model_mlp3.add(keras.layers.Dense(200, input_shape=(1000,)))
model_mlp3.add(keras.layers.Activation('relu'))

model_mlp3.add(keras.layers.Dense(300))
model_mlp3.add(keras.layers.Activation('relu'))

model_mlp3.add(keras.layers.Dense(100, activation='relu'))
model_mlp3.add(keras.layers.Dense(7, activation='sigmoid'))

model_mlp3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history3 = model_mlp3.fit(new_x_train, y_train_one_hot, epochs=25, batch_size=64, validation_data=(x_valid_vectors, y_valid_one_hot))

plt.plot(history3.history['acc'])
plt.plot(history3.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

y_pred2_3 = model_mlp3.predict_classes(x_test_vectors)

classification_report(y_test, y_pred2_3, target_names=["class 1", "class 2", "class 3", "class 4", "class 5", "class 6"])

print(precision_score(y_test, y_pred2_3, average='macro'))

print(recall_score(y_test, y_pred2_3, average='macro'))

cm2_3 = confusion_matrix(y_pred=y_pred2_3, y_true=y_test)

plot_confusion_matrix(cm           = cm2_3, 
                      normalize    = False,
                      target_names = ["class 1", "class 2", "class 3", "class 4", "class 5", "class 6"],
                      title        = "Confusion Matrix")

print('Accuracy = {}'.format(np.sum(y_pred2_3 == y_test) / len(y_test)))


y_valid_pred = model_mlp3.predict_classes(x_valid_vectors)


cm4 = confusion_matrix(y_pred=y_valid_pred, y_true=y_valid)

plot_confusion_matrix(cm           = cm4, 
                      normalize    = False,
                      target_names = ["class 1", "class 2", "class 3", "class 4", "class 5", "class 6"],
                      title        = "Confusion Matrix")

temp = []
for i in range(len(x_valid)):
    if y_valid_pred.tolist()[i] == 6 and y_valid.tolist()[i]==1:
        temp.append(i)

temp2 = []
for i in range(len(x_valid)):
    if y_valid_pred.tolist()[i] == 1 and y_valid.tolist()[i]==6:
        temp2.append(i)

print(len(temp))
print(len(temp2))

ind = 9
x_valid[temp[ind]], y_valid_pred.tolist()[temp[ind]], y_valid.tolist()[temp[ind]]

ind = 11
x_valid[temp2[ind]], y_valid_pred.tolist()[temp2[ind]], y_valid.tolist()[temp2[ind]]


