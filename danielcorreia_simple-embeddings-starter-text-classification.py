import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plot figures in the notebook wihtout the need to call plt.show()
%matplotlib inline 
plt.style.use("seaborn-ticks") #set default plotting style for matplotlib

import time
import os
print(os.listdir("../input")) #Print directories/folders in the directory: current_working_directory/input/embeddings
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

train.head()
test.head()
print(train.shape,train.qid.nunique())
print(test.shape,test.qid.nunique())
from tensorflow.keras.preprocessing.text import Tokenizer
print(Tokenizer.__doc__)
num_possible_tokens=10000 #At this stage this was chosen arbitrarily

tokenizer=Tokenizer(num_words=num_possible_tokens) #Instantiate tokenizer class with number of possible tokens
tokenizer.fit_on_texts(train.question_text) #Fit the tokenizer to training data
sequences_train=tokenizer.texts_to_sequences(train.question_text) #Convert training data to vectors
sequences_test=tokenizer.texts_to_sequences(test.question_text) #Convert test data to vectors
sequences_train[0:5]
max_len=np.max([len(i) for i in sequences_train]+[len(i) for i in sequences_test])
print(max_len)
from tensorflow.keras.preprocessing.sequence import pad_sequences
print(pad_sequences.__doc__)
X=pad_sequences(sequences_train,maxlen=max_len) #Pad the training data, later to be split into a smaller training set and a validation set
X_test=pad_sequences(sequences_test,maxlen=max_len) #Pad the test data

y=train.target.values #Make and independent target variable from the training target. Also to be split into a smaller training set and validation set.

print(X[0:10,:]) # Print first ten rows of the training data
print(X.shape)
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y, test_size=0.2, random_state=42)
import tensorflow.keras as keras
from tensorflow.keras import layers

print(layers.Embedding.__doc__)
embedding_dimension=32 # Arbitraily choose an embedding dimension,the 157 dimension input vector will be compressed down to this dimension

model=keras.models.Sequential() # Instantiate the Sequential class

model.add(layers.Embedding(num_possible_tokens+1,embedding_dimension,input_length=max_len)) # Creat embedding layer as described above
model.add(layers.Flatten()) #Flatten the embedding layer as input to a Dense layer
model.add(layers.Dense(32, activation='relu')) # Dense layer with relu activation
model.add(layers.Dense(1,activation='sigmoid')) # Dense layer with sigmoid activation for binary target
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy']) #binary cross entropy is used as the loss function and accuracy as the metric 
model.summary() # print out summary of the network
batch_size=1024 # Choose a batch size
epochs=3 #Choose number of epochs to train

history=model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=[X_val,y_val])
from sklearn.metrics import f1_score

val_pred=model.predict(X_val,batch_size=batch_size).ravel() # predict the values in the validation set which the neural net has not seen

f1_score(y_val,val_pred>0.5) #Predict the f1 score at a threshold of 50%, the point at which our binary target is split in our neural networks output probability distribution
Threshold=[] # List ot store tested thresholds
f1=[] # List to store associated f1 score for threshold

for i in np.arange(0.1, 0.501, 0.01):
    Threshold.append(i)
    temp_val_pred=val_pred>i # convert to True or False Boolean based on threshold
    temp_val_pred=temp_val_pred.astype(int) # Convert Boolean to integer
    score=f1_score(y_val,temp_val_pred) #Calculate f1 score at threshold
    f1.append(score) #store f1 score
    print("Threshold: {} \t F1 Score: {}".format(np.round(i,2),score))
best_threshold=Threshold[np.argmax(f1)] #Get threshold at index of largest f1 score.
best_threshold
test_pred=model.predict(X_test,batch_size=4096).ravel() #Predict test data

df=pd.DataFrame({'qid':test.qid.values,'prediction':test_pred}) #Create dataframe of unique id's and predicted target 
df.prediction=(df.prediction>best_threshold).astype(int) #Convert target to binary based on best f1 threshold
df.head()
df.to_csv("submission.csv", index=False)