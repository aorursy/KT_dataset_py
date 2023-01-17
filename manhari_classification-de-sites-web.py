# Librairies utilisées

import os
import numpy as np
import pandas as pd

from keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Embedding
from keras.models import Sequential
from sklearn import metrics, model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import Input, Model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Récupération du dataset
df = pd.read_csv("../input/web-classification/final_cleaned_dataset.csv")

# Élimination des lignes à contenu vide
df = df[df['content'].notnull()]

# Élimination de la colonne inutile
del df['Domaine.1']

# Récupération des domaines et du contenu
tmp = df[['Domaine', 'content']]

# Transformation de la colonne content en listes
x = tmp.apply(lambda row: str(row.content.split()), axis=1)

# Récupération des catégories
y = df['Id']

# Transformation des catégories en suite ordonnée
y = y.replace([46,1237,49,23,4,1243,37,35,53,1000], range(0,10))

# Séparation des training sets (75%) et testing sets (25%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# Transformation TF-IDF
vect = TfidfVectorizer(max_features=75000)
x_train_tfidf = vect.fit_transform(x_train).toarray()
x_test_tfidf = vect.transform(x_test).toarray()
# Création du réseau de neurones
model = Sequential()
node = 512
nLayers = 4
shape = x_train_tfidf.shape[1]
nClasses = 10
dropout = 0.5

model.add(Dense(node,input_dim=shape, activation='relu'))
model.add(Dropout(dropout))
for i in range(0, nLayers):
    model.add(Dense(node,input_dim=node, activation='relu'))
    model.add(Dropout(dropout))
model.add(Dense(nClasses, activation='softmax'))

# Compilation du modèle
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrainement du modèle
hist = model.fit(x_train_tfidf, y_train, validation_data=(x_test_tfidf, y_test), epochs=10, batch_size=10)
# Évaluation du modèle
loss, accuracy = model.evaluate(x_train_tfidf, y_train, verbose=False)
print("Training accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test_tfidf, y_test, verbose=False)
print("Testing accuracy:  {:.4f}".format(accuracy))

# Données
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
n = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

# Graphe de précision
plt.subplot(1, 2, 1)
plt.plot(n, acc, 'b', label='Training accuracy')
plt.plot(n, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

# Graphe de perte
plt.subplot(1, 2, 2)
plt.plot(n, loss, 'b', label='Training loss')
plt.plot(n, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
# Récupération du dataset
df = pd.read_csv("../input/web-classification/final_cleaned_dataset.csv")

# Élimination des lignes à contenu vide
df = df[df['content'].notnull()]

# Élimination de la colonne inutile
del df['Domaine.1']

# Récupération des domaines et du contenu
tmp = df[['Domaine', 'content']]

# Transformation de la colonne content en listes
x = tmp.apply(lambda row: str(row.content.split()), axis=1)

# Récupération des catégories
y = df['Id']

# Vectorisation
matrix = CountVectorizer(max_features=75000)
a = matrix.fit_transform(x).toarray()

# Séparation des training sets (75%) et testing sets (25%)
x_train2, x_test2, y_train2, y_test2 = train_test_split(a, y, test_size=0.25)
# Classification naïve bayésienne (gaussienne)
g = GaussianNB()
g.fit(x_train2, y_train2)
y_pred = g.predict(x_test2)
# Précision
print("Accuracy:", metrics.accuracy_score(y_test2, y_pred))
# Récupération du dataset
df = pd.read_csv("../input/web-classification/final_cleaned_dataset.csv")

# Élimination des lignes à contenu vide
df = df[df['content'].notnull()]

# Élimination de la colonne inutile
del df['Domaine.1']

# Récupération des domaines et du contenu
tmp = df[['Domaine', 'content']]

# Transformation de la colonne content en listes
x = tmp.apply(lambda row: str(row.content.split()), axis=1)

# Récupération des catégories
y = df['Id']

# Séparation des training sets (75%) et testing sets (25%)
x_train2bis, x_test2bis, y_train2bis, y_test2bis = train_test_split(x, y, test_size=0.25)
# Probabilité de chaque catégorie
prob = np.array(y_train2bis.value_counts()/len(y_train2))
print(prob)

# Transformation en tokens (fréquences des mots)
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(x_train2bis)

x_train_count =  count_vect.transform(x_train2bis).todense()
x_test_count =  count_vect.transform(x_test2bis).todense()
# Probabilité de chaque mot par classe
wordFreq = pd.DataFrame(columns=['words','class0','class1','class2','class3','class4','class5','class6','class7','class8','class9'])
wordFreq['words'] = count_vect.get_feature_names()
x_train_class0 = x_train_count[y_train2==0]
x_train_class1 = x_train_count[y_train2==1]
x_train_class2 = x_train_count[y_train2==2]
x_train_class3 = x_train_count[y_train2==3]
x_train_class4 = x_train_count[y_train2==4]
x_train_class5 = x_train_count[y_train2==5]
x_train_class6 = x_train_count[y_train2==6]
x_train_class7 = x_train_count[y_train2==7]
x_train_class8 = x_train_count[y_train2==8]
x_train_class9 = x_train_count[y_train2==9]

count_class0 = np.sum(x_train_class0,axis=0)
count_class1 = np.sum(x_train_class1,axis=0)
count_class2 = np.sum(x_train_class2,axis=0)
count_class3 = np.sum(x_train_class3,axis=0)
count_class4 = np.sum(x_train_class4,axis=0)
count_class5 = np.sum(x_train_class5,axis=0)
count_class6 = np.sum(x_train_class6,axis=0)
count_class7 = np.sum(x_train_class7,axis=0)
count_class8 = np.sum(x_train_class8,axis=0)
count_class9 = np.sum(x_train_class9,axis=0)

vocab_size0 = len(np.where(count_class0==0)[1])
vocab_size1 = len(np.where(count_class1==0)[1])
vocab_size2 = len(np.where(count_class2==0)[1])
vocab_size3 = len(np.where(count_class3==0)[1])
vocab_size4 = len(np.where(count_class4==0)[1])
vocab_size5 = len(np.where(count_class5==0)[1])
vocab_size6 = len(np.where(count_class6==0)[1])
vocab_size7 = len(np.where(count_class7==0)[1])
vocab_size8 = len(np.where(count_class8==0)[1])
vocab_size9 = len(np.where(count_class9==0)[1])

alpha=10
count_class0 = np.array((count_class0+alpha)/(np.sum(count_class0)+vocab_size0+1))
count_class1 = np.array((count_class1+alpha)/(np.sum(count_class1)+vocab_size1+1))
count_class2 = np.array((count_class2+alpha)/(np.sum(count_class2)+vocab_size2+1))
count_class3 = np.array((count_class3+alpha)/(np.sum(count_class3)+vocab_size3+1))
count_class4 = np.array((count_class4+alpha)/(np.sum(count_class4)+vocab_size4+1))
count_class5 = np.array((count_class5+alpha)/(np.sum(count_class5)+vocab_size5+1))
count_class6 = np.array((count_class6+alpha)/(np.sum(count_class6)+vocab_size6+1))
count_class7 = np.array((count_class7+alpha)/(np.sum(count_class7)+vocab_size7+1))
count_class8 = np.array((count_class8+alpha)/(np.sum(count_class8)+vocab_size8+1))
count_class9 = np.array((count_class9+alpha)/(np.sum(count_class9)+vocab_size9+1))

wordFreq['class0'] = pd.Series(count_class0.ravel())
wordFreq['class1'] = pd.Series(count_class1.ravel())
wordFreq['class2'] = pd.Series(count_class2.ravel())
wordFreq['class3'] = pd.Series(count_class3.ravel())
wordFreq['class4'] = pd.Series(count_class4.ravel())
wordFreq['class5'] = pd.Series(count_class5.ravel())
wordFreq['class6'] = pd.Series(count_class6.ravel())
wordFreq['class7'] = pd.Series(count_class7.ravel())
wordFreq['class8'] = pd.Series(count_class8.ravel())
wordFreq['class9'] = pd.Series(count_class9.ravel())
# Calcul de la précision d'entrainement
train_preds = np.zeros(len(x_train_count))

for i in range(len(x_train_count)):
    idx = np.where(x_train_count[i,:]!=0)[1]
    lh0 = wordFreq['class0'].iloc[idx].prod()
    lh1 = wordFreq['class1'].iloc[idx].prod()
    lh2 = wordFreq['class2'].iloc[idx].prod()
    lh3 = wordFreq['class3'].iloc[idx].prod()
    lh4 = wordFreq['class4'].iloc[idx].prod()
    lh5 = wordFreq['class5'].iloc[idx].prod()
    lh6 = wordFreq['class6'].iloc[idx].prod()
    lh7 = wordFreq['class7'].iloc[idx].prod()
    lh8 = wordFreq['class8'].iloc[idx].prod()
    lh9 = wordFreq['class9'].iloc[idx].prod()
    
    p = []
    p.clear()
    p.append(lh0*prob[0])
    p.append(lh1*prob[1])
    p.append(lh2*prob[2])
    p.append(lh3*prob[3])
    p.append(lh4*prob[4])
    p.append(lh5*prob[5])
    p.append(lh6*prob[6])
    p.append(lh7*prob[7])
    p.append(lh8*prob[8])
    p.append(lh9*prob[9])
    
    train_preds[i] = p.index(max(p))

matches = np.sum(y_train2==train_preds)
print('Accuracy: '+str(matches/len(train_preds)))
# Calcul de la précision de test
valid_preds = np.zeros(len(x_test_count))
for i in range(len(x_test_count)):
    idx = np.where(x_test_count[i,:]!=0)[1]
    lh0 = wordFreq['class0'].iloc[idx].prod()
    lh1 = wordFreq['class1'].iloc[idx].prod()
    lh2 = wordFreq['class2'].iloc[idx].prod()
    lh3 = wordFreq['class3'].iloc[idx].prod()
    lh4 = wordFreq['class4'].iloc[idx].prod()
    lh5 = wordFreq['class5'].iloc[idx].prod()
    lh6 = wordFreq['class6'].iloc[idx].prod()
    lh7 = wordFreq['class7'].iloc[idx].prod()
    lh8 = wordFreq['class8'].iloc[idx].prod()
    lh9 = wordFreq['class9'].iloc[idx].prod()
    
    p = []
    p.clear()
    p.append(lh0*prob[0])
    p.append(lh1*prob[1])
    p.append(lh2*prob[2])
    p.append(lh3*prob[3])
    p.append(lh4*prob[4])
    p.append(lh5*prob[5])
    p.append(lh6*prob[6])
    p.append(lh7*prob[7])
    p.append(lh8*prob[8])
    p.append(lh9*prob[9])
    
    valid_preds[i] = p.index(max(p))

matches = np.sum(y_test2==valid_preds)
print('Accuracy: '+str(matches/len(valid_preds)))
# Récupération du dataset
df = pd.read_csv("../input/web-classification/final_cleaned_dataset.csv")

# Élimination des lignes à contenu vide
df = df[df['content'].notnull()]

# Élimination de la colonne inutile
del df['Domaine.1']

# Récupération des domaines et du contenu
tmp = df[['Domaine', 'content']]

# Transformation de la colonne content en listes
x = tmp.apply(lambda row: str(row.content.split()), axis=1)

# Récupération des catégories
y = df['Id']

# Transformation des catégories en suite ordonnée
y = y.replace([46,1237,49,23,4,1243,37,35,53,1000], range(0,10))
# Transformation en tenseurs
tokenizer = Tokenizer(num_words=75000)
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)

word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=200)
labels = to_categorical(np.asarray(y))
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Séparation des training sets (75%) et testing sets (25%)
x_train3, x_test3, y_train3, y_test3 = train_test_split(data, labels, test_size=0.25)
# Utilisation des vecteurs globaux pour mots
embeddings_index = {}
f = open(os.path.join('../input/glove-global-vectors-for-word-representation/', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
# Matrice d'intégration
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# Couche d'intégration
embedding_layer = Embedding(len(word_index) + 1, 100, input_length=200)
# Réseau de neurones convolutif
sequence_input = Input(shape=(200,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
a = Conv1D(128, 5, activation='relu')(embedded_sequences)
a = MaxPooling1D(5)(a)
a = Conv1D(128, 5, activation='relu')(a)
a = MaxPooling1D(5)(a)
a = Conv1D(128, 5, activation='relu')(a)
a = MaxPooling1D(3)(a)
a = Flatten()(a)
a = Dense(128, activation='relu')(a)
preds = Dense(10, activation='softmax')(a)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Entrainement
hist2 = model.fit(x_train3, y_train3, validation_data=(x_test3, y_test3), epochs=10, batch_size=64)
# Évaluation du modèle
loss, accuracy = model.evaluate(x_train3, y_train3, verbose=False)
print("Training accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test3, y_test3, verbose=False)
print("Testing accuracy:  {:.4f}".format(accuracy))

# Données
acc = hist2.history['accuracy']
val_acc = hist2.history['val_accuracy']
loss = hist2.history['loss']
val_loss = hist2.history['val_loss']
n = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

# Graphe de précision
plt.subplot(1, 2, 1)
plt.plot(n, acc, 'b', label='Training accuracy')
plt.plot(n, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

# Graphe de perte
plt.subplot(1, 2, 2)
plt.plot(n, loss, 'b', label='Training loss')
plt.plot(n, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
# Récupération du dataset
df = pd.read_csv("../input/web-classification/final_cleaned_dataset.csv")

# Élimination des lignes à contenu vide
df = df[df['content'].notnull()]

# Élimination de la colonne inutile
del df['Domaine.1']

# Récupération des domaines et du contenu
tmp = df[['Domaine', 'content']]

# Transformation de la colonne content en listes
x = tmp.apply(lambda row: str(row.content.split()), axis=1)

# Récupération des catégories
y = df['Id']

# Séparation des training sets (75%) et testing sets (25%)
x_train4, x_test4, y_train4, y_test4 = train_test_split(x, y, test_size=0.25)
# Vectorisation
vectorizer = CountVectorizer()
vectorizer.fit(x_train4)

x_train_vect = vectorizer.transform(x_train4)
x_test_vect = vectorizer.transform(x_test4)
# Régression logistique
classifier = LogisticRegression(solver='liblinear', max_iter=500)
classifier.fit(x_train_vect, y_train4)

# Précision
score = classifier.score(x_test_vect, y_test4)
print("Accuracy:", score)