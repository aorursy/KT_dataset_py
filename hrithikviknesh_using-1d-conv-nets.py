import sklearn
from sklearn.metrics import roc_auc_score
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Input,MaxPooling1D,GlobalMaxPooling1D,Conv1D,Embedding
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
train=pd.read_csv("train.csv",low_memory=False)
train
test=pd.read_csv("test.csv",low_memory=False)
test
MAX_SEQ_LENGTH=100
MAX_VOCAB_SIZE=20000 #This is the maximum number of unique words that will be tokenized
EMBEDDING_DIM=100 # Each word will be represented as 100 dim vector\
VALIDATION_SPLIT=0.2 #Useful while training
BATCH_SIZE=128
EPOCHS=10
word2vec={}
with open(os.path.join("../large_data/glove.6B/glove.6B.%sd.txt" % EMBEDDING_DIM),encoding="utf-8") as f:
    for line in f:
        values=line.split()
        word=values[0]
        embed=np.asarray(values[1:],dtype="float32")
        word2vec[word]=embed
print("Found ",len(word2vec)," word vectors")   
sentences=train["comment_text"].fillna("DUMMY_VALUE").values   #.values returns a numpy array
possible_labels=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
targets=train[possible_labels].values # returns a one hot encoded label vector for each example in train data
targets.shape
tokenizer=Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences=tokenizer.texts_to_sequences(sentences)
word2idx=tokenizer.word_index
print("Found %s unique tokens " % len(word2idx))
data=pad_sequences(sequences,maxlen=MAX_SEQ_LENGTH) # padding is pre by default
print("shape of data is ",data.shape)
word2idx
num_words=min(MAX_VOCAB_SIZE,len(word2idx)+1) # Num of words should be less than or equal to MAX_VOCAB_SIZE

# The +1 term indicates that the tokenizer indexing begins from 1

embedding_matrix=np.zeros((num_words,EMBEDDING_DIM))
for word,pos_from_start in word2idx.items():
    if pos_from_start<MAX_VOCAB_SIZE:
        embedding_vector=word2vec.get(word) #we use get method instead of indexing because it helps if the word is not present in the dictionary
        if embedding_vector is not None:
            embedding_matrix[pos_from_start]=embedding_vector
embedding_matrix.shape
embedding_layer=Embedding(num_words,
                          EMBEDDING_DIM,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQ_LENGTH,
                          trainable=False
                         )
input_=Input(shape=(MAX_SEQ_LENGTH,))
x=embedding_layer(input_)
x=Conv1D(128,3,activation="relu")(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation="relu")(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation="relu")(x)
x=GlobalMaxPooling1D()(x) # max from every input channel
# it also indicates which timestep value in that sequnce was most influential for classification
x=Dense(128,activation="relu")(x)
output=Dense(len(possible_labels),activation="sigmoid")(x)
# we use sigmoid classifier so that each of the 6 units in the last layer act as a linear classifier(y/n)

model1=Model(input_,output)
model1.compile(loss="binary_crossentropy",
             optimizer="rmsprop",
             metrics=["accuracy"]
              )
history=model1.fit(
                data,
                targets,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_split=VALIDATION_SPLIT
                )
def plot_curves(history):
    fig,(ax0,ax1)=plt.subplots(2,1,figsize=(8,8))
    ax0.plot(history.history["loss"],label="loss")
    ax0.plot(history.history["val_loss"],label="val_loss")
    ax0.legend()
    ax1.plot(history.history["accuracy"],label="accuracy")
    ax1.plot(history.history["val_accuracy"],label="val_accuracy")
    ax1.legend()
    plt.show()
plot_curves(history)
model2=Model(input_,output)
model2.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"]
              )
history=model2.fit(
                data,
                targets,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_split=VALIDATION_SPLIT
                )
plot_curves(history)
input_=Input(shape=(MAX_SEQ_LENGTH,))
x=embedding_layer(input_)
x=Conv1D(128,3,activation="relu")(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation="relu")(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation="relu")(x)
x=GlobalMaxPooling1D()(x) # max from every input channel
# it also indicates which timestep value in that sequnce was most influential for classification
x=Dense(128,activation="relu")(x)
x=tf.keras.layers.Dropout(0.3)(x)
output=Dense(len(possible_labels),activation="sigmoid")(x)
# we use sigmoid classifier so that each of the 6 units in the last layer act as a linear classifier(y/n)

model3=Model(input_,output)
model3.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"]
              )
history=model3.fit(
                data,
                targets,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_split=VALIDATION_SPLIT
                )
plot_curves(history)
early_stopping=tf.keras.callbacks.EarlyStopping(patience=5,monitor="val_accuracy")
model_checkpoint=tf.keras.callbacks.ModelCheckpoint("model3.h5",monitor="val_accuracy",save_best_only=True)
EPOCHS=100
history=model3.fit(
                data,
                targets,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_split=VALIDATION_SPLIT,
                callbacks=[early_stopping,model_checkpoint]
                )
model=tf.keras.models.load_model("model3.h5")
p=model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))
p.shape
test_sentences=test["comment_text"].fillna("DUMMY_VALUE").values
test_sequences=tokenizer.texts_to_sequences(test_sentences)
test_data=pad_sequences(test_sequences,maxlen=MAX_SEQ_LENGTH)
pred=model.predict(test_data)
pred[:,0].shape
possible_labels
submit1=pd.DataFrame(columns=["id","toxic","severe_toxic","threat","insult","identity_hate"])
submit1["id"]=test["id"]
i=0
for col in possible_labels:
    submit1[col]=pred[:,i]
    i=i+1
submit1
submit1.index = submit1.index+1
submit1
submit1.to_csv("submission1.csv",index=False)
a=pd.read_csv("submission1.csv")
a
