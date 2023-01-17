# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import nltk
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_df.head()
nltk.download('stopwords')
from nltk.corpus import stopwords
X = train_df['text']
y = train_df['target']
# Importing word_tokenize to tokenize the text before processing
import re 
nltk.download('punkt')
from nltk.tokenize import word_tokenize
replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
bad_symbols_re = re.compile('[^0-9a-z #+_]')
links_re = re.compile('(www|http)\S+')

Stopwords = set(stopwords.words('english'))
Stopwords.remove('no')
Stopwords.remove('not')

lemmatizer = nltk.stem.WordNetLemmatizer()
def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    
    text = text.lower()  # lowercase text
    text = re.sub(replace_by_space_re," ",text) # replace symbols by space
    text = re.sub(bad_symbols_re, "",text) # remove bad symbols
    text = re.sub(links_re, "",text) # remove hyperlinks
    
    word_tokens = word_tokenize(text) # Creating word tokens out of the text
    
    filtered_tokens=[]
    for word in word_tokens:
        if word not in Stopwords:
            filtered_tokens.append(lemmatizer.lemmatize(word))
    
    text = " ".join(word for word in filtered_tokens)
    return text
X = [text_prepare(x) for x in X]
import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import Embedding # For creating word embeddings
from tensorflow.keras.preprocessing.sequence import pad_sequences # for paddding the inputs to fixed length
from tensorflow.keras.models import Sequential # Sequential model
from tensorflow.keras.preprocessing.text import one_hot # Converting words to one-hot representation
from tensorflow.keras.layers import LSTM # LSTM layer
from tensorflow.keras.layers import Dense # Dense layer of one node for probability of outcome
## Vocabulary size
voc_size =5000
onehot_X = [one_hot(words,voc_size) for words in X]
onehot_X
sent_length =40
embedded_docs_X = pad_sequences(onehot_X,padding='pre',maxlen=sent_length)
print(embedded_docs_X)
from tensorflow.keras.layers import Dropout
## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
X_final = np.array(embedded_docs_X)
y_final = np.array(y)
X_final.shape, y_final.shape
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X_final,y_final,test_size=0.33,random_state=42)
model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=15,batch_size=64)
from sklearn.metrics import confusion_matrix, accuracy_score
y_val_pred = model.predict_classes(X_val)
y_val_pred
accuracy_score(y_val,y_val_pred)
