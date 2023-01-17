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
import matplotlib.pyplot as pt
import seaborn as sns
### Change this dataset directory to you own dataset directory
data = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

data.head()
data.shape
sns.countplot(data['sentiment'])
data.info()
data.describe()
data['sentiment'] = data['sentiment'].map({'positive':1, 'negative':0})
data = data.drop_duplicates()
data = data.sample(frac = 0.40345)
X = data.drop('sentiment', axis = 1)
y = data['sentiment']
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

## Defining Vocabulary Size
### Hyperparameter, you can change it according to yourself.
voc_size = 5000
inputs = X.copy()
inputs.reset_index(inplace = True)
inputs.head()
import nltk
import re
from nltk.corpus import stopwords
## Data Preprocessing

'''
Kindly keep Patience while executing this cell, 
    This cell may require running time of approx 10-20 minutes.
    
'''
from nltk.stem import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(len(inputs)):
    review = re.sub('[^0-9a-zA-Z]', ' ',inputs['review'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    
    corpus.append(review)
corpus = np.array(corpus)
encoded_corpus = np.array([one_hot(words, voc_size) for words in corpus])
### Another Hyperparameter
max_sen_length = 250
embedded_corpus = pad_sequences(encoded_corpus, maxlen = max_sen_length, padding = 'pre')

### Modelling
embedding_vec_feat = 80
model = Sequential()
model.add(Embedding(voc_size, embedding_vec_feat, input_length=max_sen_length))
model.add(Dropout(0.7))
model.add(LSTM(150))
model.add(Dropout(0.7))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
X = embedded_corpus.copy()
y = np.array(y)
np.savez("LSTMinputV1", inputs = X, target = y)
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X, y, 
                                                     test_size = 0.33, 
                                                     random_state = 0)
model.fit(x_train, y_train, 
         validation_data=(x_valid, y_valid), 
         epochs = 8, 
         batch_size = 64)
score = model.history.history
pt.title("Accuracy vs Epoch")
pt.xlabel("No. of Epochs")
pt.ylabel('Accuracy')
pt.plot(model.history.epoch, score['accuracy'], c = 'blue', label = 'Training Accuracy')
pt.plot(model.history.epoch, score['val_accuracy'], c = 'orange', label = 'Validation Accuracy')
pt.legend()
pt.show()
pt.title("Loss vs Epoch")
pt.xlabel("No. of Epochs")
pt.ylabel("Loss")
pt.plot(model.history.epoch, score['loss'], c = 'green', label = 'Training loss')
pt.plot(model.history.epoch, score['val_loss'], c = 'red', label = 'Validation loss')
pt.legend()
pt.show()
model.save("modelv"+str(i)+'.h5')
i+=1
