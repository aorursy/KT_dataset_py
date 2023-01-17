# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
folder = '../input/'
train = pd.read_csv(folder+"train.csv")

test = pd.read_csv(folder+"test.csv")

test_labels = pd.read_csv(folder+"test_labels.csv")

submission = pd.read_csv(folder+"sample_submission.csv")
train.head(10)
# Cantidad de observaciones

train.shape
# Defino y (Salida del modelo)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

print(y.shape)

print(y[:10])
# Dataset muy desbalanceado

toxic_ratio = (y.sum(axis = 1) > 0).sum()/y.shape[0]

print('Porcentaje de comentarios toxicos:', toxic_ratio)
# La mayoría son toxic

print(train[list_classes].sum())

print()

print(train[list_classes].sum()/y.shape[0])
# Superposición entre las clases (Multilabel)

for cl in list_classes[1:]:

    N = ((train['toxic'] == 0) & (train[cl] == 1)).sum()

    print(f'Es {cl} pero no es toxic:', N)
# Baseline (Suponer que siempre elijo zeros (No toxico))

1-(train[list_classes].sum().values/len(train)).mean()
((y == np.zeros_like(y)).sum(axis=0)/len(y)).mean()
train[list_classes].sum().values/len(train)
X_train, X_valid, Y_train, Y_valid = train_test_split(train['comment_text'], y, test_size = 0.1)



print(X_train.shape, X_valid.shape)

print(Y_train.shape, Y_valid.shape)
X_train[:10]
raw_text_train = X_train.apply(str.lower)

raw_text_valid = X_valid.apply(str.lower)

raw_text_test = test["comment_text"].apply(str.lower)
print(raw_text_train[:10]) # Recordar que train_test_split hace shuffle 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



max_features = 10000



tfidf_vectorizer = TfidfVectorizer(max_df=0.11, min_df=1,

                                   max_features=max_features,

                                   stop_words='english')



%time tfidf_matrix_train = tfidf_vectorizer.fit_transform(raw_text_train)
%time tfidf_matrix_valid = tfidf_vectorizer.transform(raw_text_valid)
count_vectorizer = CountVectorizer(max_df=0.11, min_df=1,

                                   max_features=max_features,

                                   stop_words='english')



%time count_matrix_train = count_vectorizer.fit_transform(raw_text_train)
tfidf_matrix_train.shape, count_matrix_train.shape
sparsity = 1 - (tfidf_matrix_train>0).sum()/(tfidf_matrix_train.shape[0]*tfidf_matrix_train.shape[1])

print(sparsity)
top_10 = np.argsort(tfidf_matrix_train.sum(axis=0))[0,::-1][0,:10].tolist()[0]

feature_names = np.array(tfidf_vectorizer.get_feature_names())

print(feature_names[np.array(top_10)])
top_10_count = np.argsort(count_matrix_train.sum(axis=0))[0,::-1][0,:10].tolist()[0]

feature_names_count = np.array(count_vectorizer.get_feature_names())

print(feature_names_count[np.array(top_10_count)])
dense_matrix_train = tfidf_matrix_train.todense()

dense_matrix_valid = tfidf_matrix_valid.todense()
from keras.models import Sequential

from keras.layers import Dense

from keras import initializers
input_features = dense_matrix_train.shape[1]

output_size = Y_train.shape[1]



model_rl = Sequential()

model_rl.add(Dense(output_size, input_dim=input_features, activation='sigmoid', 

                   kernel_initializer=initializers.normal(mean=0, stddev=0.001)))

model_rl.summary()

model_rl.compile('Adam', loss='binary_crossentropy', metrics=['accuracy'])
model_rl.evaluate(dense_matrix_valid, Y_valid)
batch_size = 128

epochs = 20

model_rl.fit(dense_matrix_train, 

          Y_train, 

          batch_size = batch_size,

          epochs=epochs, 

          verbose=1, 

          validation_data=(dense_matrix_valid, Y_valid))
(model_rl.get_weights()[0]).shape
salida = 3

sorted_indexes = np.argsort(model_rl.get_weights()[0][:,salida])[::-1]

np.array(tfidf_vectorizer.get_feature_names())[sorted_indexes][:20]
from keras import regularizers

from keras import initializers

from keras.layers import Activation

from keras import optimizers
default_initializer = initializers.normal(mean=0, stddev=0.01)

input_features = dense_matrix_train.shape[1]

output_size = Y_train.shape[1]

hidden_units = 100

lambd = 0 #0.001

model_sig_nn = Sequential()

model_sig_nn.add(Dense(200,

                       input_dim=input_features, 

                       kernel_regularizer=regularizers.l2(lambd), 

                       kernel_initializer=default_initializer,

                       name="Capa_Oculta_1"))

model_sig_nn.add(Activation('sigmoid'))

model_sig_nn.add(Dense(200,

                       input_dim=input_features, 

                       kernel_regularizer=regularizers.l2(lambd), 

                       kernel_initializer=default_initializer,

                       name="Capa_Oculta_2"))

model_sig_nn.add(Activation('sigmoid'))

model_sig_nn.add(Dense(output_size,

                       kernel_regularizer=regularizers.l2(lambd), 

                       kernel_initializer=default_initializer,

                       name="Capa_Salida"))

model_sig_nn.add(Activation('sigmoid', name="output")) 

model_sig_nn.summary()





lr = 0.001 

batch_size = 256

epochs = 10



#selectedOptimizer = optimizers.SGD(lr=lr)

selectedOptimizer = optimizers.adam(lr=lr, decay=0.001)



model_sig_nn.compile(loss = 'binary_crossentropy', optimizer=selectedOptimizer, 

                     metrics=['accuracy']) #auc
model_sig_nn.evaluate(dense_matrix_valid, Y_valid)
history = model_sig_nn.fit(dense_matrix_train, 

          Y_train, 

          batch_size = batch_size,

          epochs=epochs, 

          verbose=1, 

          validation_data=(dense_matrix_valid, Y_valid), 

         )
pred_valid = model_sig_nn.predict(dense_matrix_valid, verbose = 1)

pred_train = model_sig_nn.predict(dense_matrix_train, verbose = 1)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc

from scipy import interp

from itertools import cycle



print(roc_auc_score(Y_train, pred_train, average='macro'))

print(roc_auc_score(Y_valid, pred_valid, average='macro'))
fpr = dict()

tpr = dict()

roc_auc = dict()

n_classes = Y_valid.shape[1]

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], pred_valid[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])

    

fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), pred_valid.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
from matplotlib import pyplot as plt

# Compute macro-average ROC curve and ROC area

lw = 2

# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC

mean_tpr /= n_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves

plt.figure()

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=4)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Some extension of Receiver operating characteristic to multi-class')

plt.legend(loc="lower right")

plt.show()