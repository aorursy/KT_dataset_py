# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!unzip "/content/drive/My Drive/Analytics_Vidya_cv/av_independence_day/train1.zip"
!unzip "/content/drive/My Drive/Analytics_Vidya_cv/av_independence_day/test1.zip"
train = pd.read_csv("../input/janatahack-independence-day-2020-ml-hackathon/train.csv")
test = pd.read_csv("../input/janatahack-independence-day-2020-ml-hackathon/test.csv")
train.head()
abstracts = train['ABSTRACT']
titles = train['TITLE']
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate

import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
train.shape
filt = train['ABSTRACT'] != ""
train = train[filt]
train = train.dropna()
train['ABSTRACT'][123]
print("Computer Science:" + str(train["Computer Science"][123]))
print("Physics:" + str(train["Physics"][123]))
print("Mathematics:" + str(train["Mathematics"][123]))
print("Statistics:" + str(train["Statistics"][123]))
print("Quantitative Biology:" + str(train["Quantitative Biology"][123]))
print("Quantitative Finance:" + str(train["Quantitative Finance"][123]))
labels = train[["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology", "Quantitative Finance"]]
labels.head()
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 16
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
labels.sum(axis = 0).plot.bar()
import nltk
nltk.download('stopwords')
def preprocess_text(text):
    #remove punctuations and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    #single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    
    #removing multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    new_text = ""
    for word in text.split():
        if word not in stopwords.words("english"):
            new_text = new_text + ' ' + word
    
    return new_text
X = []
texts = list(train["ABSTRACT"])
for t in texts:
    X.append(preprocess_text(t))
y = labels.values
X[4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42, shuffle = True)
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index)+1

max_len = 200

X_train = pad_sequences(X_train, padding = 'post', maxlen = max_len)
X_test = pad_sequences(X_test, padding = 'post', maxlen = max_len)
vocab_size
from numpy import array
from numpy import asarray
from numpy import zeros
embeddings_dictionary = dict()

glove_file = open('../input/glove6b100dtxt/glove.6B.100d.txt', encoding = "utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype = 'float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()
y_test
embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
inputs = Input(shape = (max_len,))
embedding_layer = Embedding(vocab_size, 100, weights = [embedding_matrix], trainable = False)(inputs)
LSTM_1 = LSTM(256)(embedding_layer)
dense_1 = Dense(6, activation = 'sigmoid')(LSTM_1)
model = Model(inputs = inputs, outputs = dense_1)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)
history = model.fit(X_train, y_train, batch_size = 128, epochs = 20, verbose = 1)
score = model.evaluate(X_test, y_test, verbose = 1)

print("Loss : ", score[0])
print("Accuracy : ", score[1])
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
X_t = []
test_abs = list(test['ABSTRACT'])
for t in test_abs:
    X_t.append(preprocess_text(t)) 
X_t = tokenizer.texts_to_sequences(X_t)
X_t = pad_sequences(X_t, padding = 'post', maxlen = max_len)
pred = model.predict(X_t)
pred[0]
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        if pred[i][j] >= 0.45:
          pred[i][j] = 1
        else:
          pred[i][j] = 0
output = pd.DataFrame()
output['ID'] = test['ID']
output['Computer Science'] = pred[:, 0]
output['Physics'] = pred[:, 1]
output['Mathematics'] = pred[:, 2]
output['Statistics'] = pred[:, 3]
output['Quantitative Biology'] = pred[:, 4]
output['Quantitative Finance'] = pred[:, 5]
courses = [["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology", "Quantitative Finance"]]
for column in courses:
    output[column] = output[column].astype(int)
output.head()
output.to_csv("Keras LSTM.csv", index = False)


train = pd.read_csv("../input/janatahack-independence-day-2020-ml-hackathon/train.csv")
test = pd.read_csv("../input/janatahack-independence-day-2020-ml-hackathon/test.csv")
labels = train[["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology", "Quantitative Finance"]]
labels.head()
filt = train['ABSTRACT'] != ""
train = train[filt]
train = train.dropna()
import nltk
nltk.download('stopwords')
def preprocess_text(text):
    #remove punctuations and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    #single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    
    #removing multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    new_text = ""
    for word in text.split():
        if word not in stopwords.words("english"):
            new_text = new_text + ' ' + word
    
    return new_text
X = []
texts = list(train["ABSTRACT"])
for t in texts:
    X.append(preprocess_text(t))
y = labels.values
traintmp = train.drop(['TITLE'],axis=1)
df = traintmp
df['labels'] = list(zip(df['Computer Science'].tolist(), df.Physics.tolist(), df.Mathematics.tolist(), df.Statistics.tolist(),  df['Quantitative Biology'].tolist(), df['Quantitative Finance'].tolist()))
# traintmp
df1 = df.drop(["Computer Science","Physics", "Mathematics", "Statistics","Quantitative Biology", 
         "Quantitative Finance", "ID"], axis=1)
df1
from sklearn.model_selection import train_test_split
train_df, eval_df = train_test_split(df1, test_size=0.2)
!pip install simpletransformers
from simpletransformers.classification import MultiLabelClassificationModel
model = MultiLabelClassificationModel('roberta', 'roberta-base', num_labels=6, args={'train_batch_size':2, 'gradient_accumulation_steps':16, 'learning_rate': 3e-5, 'num_train_epochs': 3, 'max_seq_length': 200})
!pip install transformers==2.11.0
model.train_model(train_df)
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
X_t = []
test_abs = list(test['ABSTRACT'])
for t in test_abs:
    X_t.append(preprocess_text(t)) 
preds, outputs = model.predict(X_t)
outputs[6]
preds[6]
# df1 = df.drop(["Computer Science","Physics", "Mathematics", "Statistics","Quantitative Biology", 
#          "Quantitative Finance", "ID"], axis=1)
sub_df = pd.DataFrame(preds,columns=["Computer Science","Physics", "Mathematics", "Statistics","Quantitative Biology", 
         "Quantitative Finance"])
sub_df['ID'] = test['ID']
sub_df = sub_df[["ID","Computer Science","Physics", "Mathematics", "Statistics","Quantitative Biology", 
         "Quantitative Finance"]]
sub_df.to_csv('transformer(roberta3).csv', index=False)
