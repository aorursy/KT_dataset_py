import os
import sys
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D
from keras.layers import MaxPool1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
# just to make sure the dataset is added properly 
!ls '../input/20-newsgroup-original/20_newsgroup/20_newsgroup/'

import pandas as pd
df = pd.read_csv('../input/my-data-set/train.csv')
df['mod'] = df.TITLE+df.ABSTRACT
sentences = df['mod'].astype('str').tolist()
def topic_class(x):
    if(x['Computer Science']):
        return 0
    if(x['Physics']):
        return 1
    if(x['Mathematics']):
        return 2
    if(x['Statistics']):
        return 3
    if(x['Quantitative Biology']):
        return 4
    if(x['Quantitative Finance']):
        return 5
    
df['class']=df.apply(topic_class,axis=1)      

train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
labels_index = {'Computer Science':0,'Physics':1,'Mathematics':2,'Statistics':3,'Quantitative Biology':4,
               'Quantitative Finance':5}


train=train.append(validate)
texts=train['mod'].astype('str').tolist()

labels=train['class'].tolist()

len(texts)
len(labels_index)
# the dataset path
TEXT_DATA_DIR = r'../input/20-newsgroup-original/20_newsgroup/20_newsgroup/'
#the path for Glove embeddings
GLOVE_DIR = r'../input/glove6b/'
# make the max word length to be constant
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 1000
# the percentage of train test split to be applied
VALIDATION_SPLIT = 0.20
# the dimension of vectors to be used
EMBEDDING_DIM = 100
# filter sizes of the different conv layers 
filter_sizes = [3,4,5]
num_filters = 512
embedding_dim = 100
# dropout probability
drop = 0.5
batch_size = 30
epochs = 2
tokenizer  = Tokenizer(num_words = MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences =  tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("unique words : {}".format(len(word_index)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print(labels)
## preparing dataset


texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)
print(labels_index)

print('Found %s texts.' % len(texts))
data
len(labels_index)
len(texts)
# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
print(data)
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding = embedding_layer(inputs)

print(embedding.shape)
reshape = Reshape((MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1))(embedding)
print(reshape.shape)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=6, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('weights_cnn_sentece.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("Traning Model...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=4, verbose=1, callbacks=[checkpoint], validation_data=(x_val, y_val))

test_texts=test['mod'].astype('str').tolist()
test_sequences =  tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
x=model.predict(test_data)
x

result=[]
for out in x:
    cls=list(out).index(max(out))
    result.append(cls)
    
result    
import seaborn as sns
sns.countplot(result)
sns.countplot(test['class'])
test_data= pd.read_csv('../input/testdata/test.csv')
test_data['mod']=test_data.TITLE+test_data.ABSTRACT

test_texts=test_data['mod'].astype('str').tolist()
test_sequences =  tokenizer.texts_to_sequences(test_texts)
data_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

x=model.predict(data_test)

result=[]
for out in x:
    cls=list(out).index(max(out))
    result.append(cls)
    
result    



sns.countplot(result)
test_data.reset_index(inplace=True)
#test_data.drop('index',axis=1,inplace=True)
test_data['final_pred']=pd.Series(result)
test_data.set_index('index',inplace=True)

test_data=test_data[['ID','final_pred']]
test_data['Computer Science']=test_data.final_pred.apply(lambda x :1 if x==0 else 0)
test_data['Physics']=test_data.final_pred.apply(lambda x :1 if x==1 else 0)
test_data['Mathematics']=test_data.final_pred.apply(lambda x :1 if x==2 else 0)
test_data['Statistics']=test_data.final_pred.apply(lambda x :1 if x==3 else 0)
test_data['Quantitative Biology']=test_data.final_pred.apply(lambda x :1 if x==4 else 0)
test_data['Quantitative Finance']=test_data.final_pred.apply(lambda x :1 if x==5 else 0)

test_data.drop('final_pred',axis=1,inplace=True)

test_data.to_csv('pred3.csv',index=False)


test_data.to_csv('pred.csv',index=False)
