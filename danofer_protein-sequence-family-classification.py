import numpy as np

import pandas as pd



from keras.models import Sequential

from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten,AtrousConvolution1D,SpatialDropout1D, Dropout

from keras.layers.normalization import BatchNormalization



from keras.layers import LSTM

from keras.layers.embeddings import Embedding



# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

from keras.callbacks import EarlyStopping



from keras.preprocessing import text, sequence

from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split






# Merge the two Data set together

# Drop duplicates (unlike the original script)

df = pd.read_csv('../input/pdb_data_no_dups.csv').merge(pd.read_csv('../input/pdb_data_seq.csv'), how='inner', on='structureId').drop_duplicates(["sequence"]) # ,"classification"

# Drop rows with missing labels

df = df[[type(c) == type('') for c in df.classification.values]]

df = df[[type(c) == type('') for c in df.sequence.values]]

# select proteins

df = df[df.macromoleculeType_x == 'Protein']

df.reset_index()

print(df.shape)

df.head()
print(df.residueCount_x.quantile(0.9))

df.residueCount_x.describe()
df = df.loc[df.residueCount_x<1200]

print(df.shape[0])

df.residueCount_x.describe()
import matplotlib.pyplot as plt

from collections import Counter



# count numbers of instances per class

cnt = Counter(df.classification)

# select only K most common classes! - was 10 by default

top_classes = 10

# sort classes

sorted_classes = cnt.most_common()[:top_classes]

classes = [c[0] for c in sorted_classes]

counts = [c[1] for c in sorted_classes]

print("at least " + str(counts[-1]) + " instances per class")



# apply to dataframe

print(str(df.shape[0]) + " instances before")

df = df[[c in classes for c in df.classification]]

print(str(df.shape[0]) + " instances after")



seqs = df.sequence.values

lengths = [len(s) for s in seqs]



# visualize

fig, axarr = plt.subplots(1,2, figsize=(20,5))

axarr[0].bar(range(len(classes)), counts)

plt.sca(axarr[0])

plt.xticks(range(len(classes)), classes, rotation='vertical')

axarr[0].set_ylabel('frequency')



axarr[1].hist(lengths, bins=50, normed=False)

axarr[1].set_xlabel('sequence length')

axarr[1].set_ylabel('# sequences')

plt.show()
from sklearn.preprocessing import LabelBinarizer



# Transform labels to one-hot

lb = LabelBinarizer()

Y = lb.fit_transform(df.classification)


# maximum length of sequence, everything afterwards is discarded! Default 256

# max_length = 256

max_length = 300



#create and fit tokenizer

tokenizer = Tokenizer(char_level=True)

tokenizer.fit_on_texts(seqs)

#represent input data as word rank number sequences

X = tokenizer.texts_to_sequences(seqs)

X = sequence.pad_sequences(X, maxlen=max_length)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)
embedding_dim = 11 # orig 8



# create the model

model = Sequential()

model.add(Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length))

# model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')) #orig

model.add(Conv1D(filters=128, kernel_size=9, padding='valid', activation='relu',dilation_rate=1))

# model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size=2))

# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')) # orig

model.add(Conv1D(filters=64, kernel_size=4, padding='valid', activation='relu')) 

model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(200, activation='elu')) # 128

# model.add(BatchNormalization())

# model.add(Dense(64, activation='elu')) # 128

model.add(BatchNormalization())

model.add(Dense(top_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
es = EarlyStopping(monitor='val_acc', verbose=1, patience=4)



model.fit(X_train, y_train,  batch_size=128, verbose=1, validation_split=0.15,callbacks=[es],epochs=25) # epochs=15, # batch_size=128
from keras.layers import Conv1D, MaxPooling1D, Concatenate, Input

from keras.models import Sequential,Model



units = 256

num_filters = 32

filter_sizes=(3,5, 9,15,21)

conv_blocks = []



embedding_layer = Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length)

es2 = EarlyStopping(monitor='val_acc', verbose=1, patience=4)



sequence_input = Input(shape=(max_length,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)



z = Dropout(0.1)(embedded_sequences)



for sz in filter_sizes:

    conv = Conv1D(

        filters=num_filters,

        kernel_size=sz,

        padding="valid",

        activation="relu",

        strides=1)(z)

    conv = MaxPooling1D(pool_size=2)(conv)

    conv = Flatten()(conv)

    conv_blocks.append(conv)

z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(0.25)(z)

z = BatchNormalization()(z)

z = Dense(units, activation="relu")(z)

z = BatchNormalization()(z)

predictions = Dense(top_classes, activation="softmax")(z)

model2 = Model(sequence_input, predictions)

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model2.summary())



model2.fit(X_train, y_train,  batch_size=64, verbose=1, validation_split=0.15,callbacks=[es],epochs=30)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

import itertools



train_pred = model.predict(X_train)

test_pred = model.predict(X_test)

print("train-acc = " + str(accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))))

print("test-acc = " + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))))



# Compute confusion matrix

cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))



# Plot normalized confusion matrix

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

np.set_printoptions(precision=2)

plt.figure(figsize=(10,10))

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

plt.title('Confusion matrix')

plt.colorbar()

tick_marks = np.arange(len(lb.classes_))

plt.xticks(tick_marks, lb.classes_, rotation=90)

plt.yticks(tick_marks, lb.classes_)

#for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

#    plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()



print(classification_report(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1), target_names=lb.classes_))
train_pred = model2.predict(X_train)

test_pred = model2.predict(X_test)

print("train-acc = " + str(accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))))

print("test-acc = " + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))))



# Compute confusion matrix

cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))



# Plot normalized confusion matrix

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

np.set_printoptions(precision=2)

plt.figure(figsize=(10,10))

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

plt.title('Confusion matrix')

plt.colorbar()

tick_marks = np.arange(len(lb.classes_))

plt.xticks(tick_marks, lb.classes_, rotation=90)

plt.yticks(tick_marks, lb.classes_)

#for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

#    plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()



print(classification_report(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1), target_names=lb.classes_))