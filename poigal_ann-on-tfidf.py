import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
%precision %.2f

import re

# Importing required libraries
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer


# keras
from keras.models import Sequential
from keras.layers import Dense

from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
# defining function to clean text and retrive closs-validation datasets
def cleantxt(txt):
    """
    Cleans the string passed. Cleaning Includes-
    1. remove special characters/symbols
    2. convert text to lower-case
    3. retain only alphabets
    4. remove words less than 3 characters
    5. remove stop-words
    """  
    # collecting english stop words from nltk-library
    stpw = stopwords.words('english')
    
    # Adding custom stop-words
    stpw.extend(['www','http','utc'])
    stpw = set(stpw)
    
    # using regex to clean the text
    txt = re.sub(r"\n", " ", txt)
    txt = re.sub("[\<\[].*?[\>\]]", " ", txt)
    txt = txt.lower()
    txt = re.sub(r"[^a-z ]", " ", txt)
    txt = re.sub(r"\b\w{1,3}\b", " ",txt)
    txt = " ".join([x for x in txt.split() if x not in stpw])
    return txt


def load_data():
    """
    Loads data and returns train, val, and test splits
    """
    # Load the train dataset
    df = pd.read_csv("../input/train.csv")
    
    # Clean the text
    df['comment_text'] = df.comment_text.apply(lambda x : cleantxt(x))
    
    # separate explanatory and dependent variables
    X = df.iloc[:,1]
    y = df.iloc[:,2:]

    # split for cross-validation (train-60%, validation 20% and test 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=123)

    return X_train, X_val, X_test, y_train, y_val, y_test
# Load the data
X_train, X_val, X_test, y_train, y_val, y_test = load_data()
vect = TfidfVectorizer(decode_error='ignore',stop_words='english')
train_tfidf = vect.fit_transform(X_train)
val_tfidf = vect.transform(X_val)
test_tfidf = vect.transform(X_test)

ip_dim = train_tfidf.shape[1]

model = Sequential()
model.add(Dense(64, input_dim=ip_dim, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
# Compiling Model using optimizer
opt = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy',optimizer=opt)

# Fitting Model to the data
callbacks = [EarlyStopping(monitor='val_loss')]
hist_adam = model.fit(train_tfidf, np.asarray(y_train), batch_size=1000, epochs=20, verbose=2, validation_data=(val_tfidf, np.asarray(y_val)),
         callbacks=callbacks)  # starts training
#plotting Loss
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.plot(hist_adam.history['loss'], color='b', label='Training Loss')
plt.plot(hist_adam.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')
# Creating empty prediction array
col = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']

# Predict on train, val and test datasets
pred_train = model.predict(train_tfidf)
pred_test = model.predict(test_tfidf)
pred_val = model.predict(val_tfidf)

# Emply array to collect AUC scores
AUC = np.zeros((3,6))
AUC
from sklearn import metrics
for i,x in enumerate(col):
    auc = np.array([metrics.roc_auc_score(y_train[x], pred_train[:,i]),
                    metrics.roc_auc_score(y_val[x], pred_val[:,i]),
                    metrics.roc_auc_score(y_test[x], pred_test[:,i])])
    print(x,"Train AUC:",auc[0],", Val AUC:",auc[1],", Test AUC:",auc[2])
    AUC[:,i] = auc
    
avg_auc = AUC.mean(axis=1)
print("Average Train AUC:",avg_auc[0],", Average Val AUC:",avg_auc[1],", Average Test AUC:",avg_auc[2])