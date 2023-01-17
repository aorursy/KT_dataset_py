# Import the required libraries 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import json

# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
	return json.load(open(path)) 
train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')

# Text Data Features
print ("Prepare text data of Train and Test ... ")
def generate_text(data):
	text_data = [" ".join(doc['ingredients']).lower() for doc in data]
	return text_data 

train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]

# Feature Engineering 
print ("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True)
def tfidf_features(txt, flag):
    if flag == "train":
    	x = tfidf.fit_transform(txt)
    else:
	    x = tfidf.transform(txt)
    x = x.astype('float16')
    return x 

X = tfidf_features(train_text, flag="train")
X_test = tfidf_features(test_text, flag="test")

# Label Encoding - Target 
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)

## to one-hot
def to_onehot(input_):
    output_ = np.zeros((len(input_), max(input_)+1))
    for i in range(len(input_)):
        output_[i][input_[i]] = 1
    return output_
    
y = to_onehot(y)    
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Dense, Dropout, concatenate, Input, Flatten, add, Activation
from keras.callbacks import EarlyStopping
# from keras.utils import plot_model

inside_dim = 256
batch_size = 32
epochs = 20
block_n = 10

def dense_layer(x):
    x = Dense(inside_dim, input_dim=inside_dim, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)  
    return x

def dense_block(input_):
    x = input_
    y = dense_layer(x)
    y = dense_layer(y)
    x = add([x, y])
    x = Activation('relu')(x)
#     x = concatenate([x, y])
#     x = Flatten()(x)
    return x

def dense_net(input_, n):
    for _ in range(n):
        input_ = dense_block(input_)
    return input_

    
inputs = Input(shape=(3010,))    
x = Dense(inside_dim, input_dim=3010, activation="relu")(inputs)
x = BatchNormalization()(x)    
x = Dropout(rate=0.5)(x)   

x = dense_net(input_=x, n=block_n)

# model = Sequential()
# model.add(Dense(inside_dim, input_dim=3010, activation="relu"))
# model.add(BatchNormalization())
# model.add(Dropout(rate=0.5))

# model.add(Dense(inside_dim, input_dim=inside_dim, activation="relu"))
# model.add(BatchNormalization())
# model.add(Dropout(rate=0.5))
# model.add(Dense(inside_dim, input_dim=inside_dim, activation="relu"))
# model.add(BatchNormalization())
# model.add(Dropout(rate=0.5))
# model.add(Dense(inside_dim, input_dim=inside_dim, activation="relu"))
# model.add(BatchNormalization())

# ## for output
# model.add(Dense(20, input_dim=inside_dim, activation="softmax"))
outputs = Dense(20, input_dim=inside_dim, activation="softmax")(x)

model = Model(inputs=[inputs], outputs=[outputs])

model.summary()

# callback = [EarlyStopping(monitor='loss', patience=5)]

model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
# model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callback)
model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

output_name = "GPU_test_2.csv"



def onthot_to_label(input_):
    output_ = np.zeros(len(input_))
    for i in range(len(input_)):
        output_[i] = int(np.argmax(input_[i]))
    output_ = output_.astype('int8')
    return output_

# Predictions 
print ("Predict on test data ... ")
y_test = model.predict(X_test)
y_test = onthot_to_label(y_test)
y_pred = lb.inverse_transform(y_test)

# Submission
print ("Generate Submission File ... ")
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv(output_name, index=False)

sub
