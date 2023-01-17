# Importing Libraries and DATA
import numpy as np
import pandas as pd
df_dup = pd.read_csv('../input/pdb_data_no_dups.csv')
df_seq = pd.read_csv('../input/pdb_data_seq.csv')
# Merge the two Data set together
df = df_dup.merge(df_seq,how='inner',on='structureId')

# Drop rows with missing labels
df = df[[type(c) == type('') for c in df.classification.values]]
df = df[[type(c) == type('') for c in df.sequence.values]]

# select proteins
df = df[df.macromoleculeType_x == 'Protein']
df.reset_index()
df.head()
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import matplotlib.pyplot as plt
from collections import Counter

cnt = Counter(df.classification)
# select only 10 most common classes!
top_classes = 10
tmp = np.array([[c[0], c[1]] for c in cnt.most_common()[:top_classes]])
[classes, counts] = tmp[:,0], tmp[:,1].astype(int)

N = sum(counts)
plt.bar(range(len(classes)), counts/float(N))
plt.xticks(range(len(classes)), classes, rotation='vertical')
#plt.xlabel('protein class')
plt.ylabel('frequency')
plt.show()
df = df[[c in classes for c in df.classification]]
seqs = df.sequence.values

# Transform labels into one-hot
lb = LabelBinarizer()
Y = lb.fit_transform(df.classification)

lengths = [len(s) for s in seqs]
plt.hist(lengths, bins=100, normed=True)
plt.xlabel('sequence length')
plt.ylabel('frequency')
plt.show()
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

max_length = 512

#create and fit tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(seqs)
#represent input data as word rank number sequences
X = tokenizer.texts_to_sequences(seqs)
X = sequence.pad_sequences(X, maxlen=max_length)

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=.7)
X_train.shape, X_test.shape
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

embedding_dim = 16

# create the model
model = Sequential()
model.add(Embedding(len(tokenizer.index_docs)+1, embedding_dim, input_length=max_length))
model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(top_classes, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print("train-acc = " + str(accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))))
print("test-acc = " + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))))

# Compute confusion matrix
cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=lb.classes_, normalize=True)
plt.show()
