# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import string
import re
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
comp = pd.read_csv('../input/customer-bank-complaints-usa-sample/complaints1.csv', dtype=object)
samp_size = np.round((len(comp)/1.8),0).astype('int')
comp = comp.sample(samp_size, random_state= 30)
text_comp  = comp[['Consumer complaint narrative', 'Company response to consumer']]
text_comp['Money'] = np.where(text_comp['Company response to consumer'] == 'Closed with monetary relief', 1, 0)
y = text_comp['Money']
x = text_comp.drop(['Money','Company response to consumer'], axis=1)
x = x.astype(str)
new_x = x.agg(' '.join, axis=1)
# Cleaning and transforming the text to a frequency matrix of the words for each observation.
new_x = new_x.str.replace('\W', ' ') # Excluding ponctuation in the text
new_x = new_x.str.replace('nan', '') # Excluding the word 'nan'
new_x = [w.lower() for w in new_x]  # Transforming text to lowercase
stopword = set(stopwords.words('english')) 
new_x = [w for w in new_x if not w in stopword] # Excluding stopwords
txt_tok =  Tokenizer()
txt_tok.fit_on_texts(new_x)
new_x = txt_tok.texts_to_matrix(new_x, mode='count')#creating a matrix containing every single word in the text and its count in each obervation.
x_train, x_test, y_train, y_test = train_test_split(new_x , y , test_size=.1, random_state = 50)
x_train.shape # Here we can check that the dataset contains 36443 words in 106872 observations
nwords = x_train.shape[1]
# defining the network with 2 layers.
model = Sequential()
model.add(Dense(200, input_shape=(nwords,), activation='relu')) #2 Hidden layers with 200 neurons each
model.add(Dense(200, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) #Output layer. Using activation sigmoid because we want the node to give a number between 0 and 1 for classification purposes.
# compiling the network model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Fitting the network model
model.fit(x_train, y_train, epochs=20) # Fitting the model with 20 full cycles(epoch)
model.evaluate(x_test, y_test) #Evaluating the loss function and the accuracy.
#Confusion Matrix
y_pred1 =np.round(model.predict(x_test),0)
pd.DataFrame(confusion_matrix(y_test, y_pred1))
#Here we are giving up a little bit of accuracy in order to get a better preditcion for Money(1).
class_weight = {0:1, 1:5} #Picked the weights
model.fit(x_train, y_train, epochs=20, class_weight = class_weight)
#Checking the confusion table it is clear that the number of correct predictions made for 1 improved significantly.
y_pred2 =np.round(model.predict(x_test),0)
model.evaluate(x_test, y_test)
pd.DataFrame(confusion_matrix(y_test, y_pred2))
