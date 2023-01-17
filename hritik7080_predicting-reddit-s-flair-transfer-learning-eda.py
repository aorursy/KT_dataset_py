# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
import nltk
import re
posts = pd.read_csv('/kaggle/input/reddit-flair-dataset/reddit_data.csv')
posts.head()
posts['flair'].value_counts()
plt.figure(figsize=(20,8))
sns.countplot(x='flair', data=posts)
plt.title("Number of Posts of each Flair")
plt.xlabel('Flairs')
plt.ylabel("Number of Posts")
plt.show()
posts['num_comments'].sum()/posts.shape[0]
plt.figure(figsize=(20,8))
sns.distplot(posts[posts["num_comments"] < 61]["num_comments"], kde=False)
plt.grid()
plt.title("Distrbution of number of Comments on the Posts")
plt.ylabel("Number of Posts")
plt.xlabel("Number of Comments")

plt.show()
posts['score'].sum()/posts.shape[0]
plt.figure(figsize=(20,8))
sns.distplot(posts[posts["score"] < 147]["score"], kde=False)
plt.grid()
plt.title("Distrbution of Score on the Posts")
plt.ylabel("Number of Posts")
plt.xlabel("Score")

plt.show()
data_score = posts.sort_values('num_comments', ascending=False).head(15)
plt.figure(figsize=(14,7))
plt.title("Posts with highest number of Comments")
sns.barplot(y=data_score['title'],x=data_score['num_comments'])
plt.xlabel('Score',fontsize=18)
plt.ylabel("Post's Title",fontsize=18)
plt.show()
data_score = posts.sort_values('score', ascending=False).head(15)
plt.figure(figsize=(14,9))
plt.title('Posts with highest Score')
sns.barplot(y=data_score['title'],x=data_score['score'])
plt.xlabel('Score',fontsize=18)
plt.ylabel('Title',fontsize=18)
plt.show()
correlation =  posts.corr()
correlation
sns.heatmap(correlation)
plt.show()
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
flairs = posts['flair'].unique()
from wordcloud import WordCloud, STOPWORDS 
for flair in flairs:
  content = ""
  for i in posts[posts['flair']==flair]['title']:
    content+=i
  wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stop, min_font_size = 10).generate(content)
  plt.figure(figsize = (8, 8), facecolor = None) 
  plt.imshow(wordcloud) 
  plt.title("WordClound for "+flair)
  plt.axis("off")
  plt.tight_layout(pad = 0) 
  plt.show() 
nltk.download('wordnet')

def remove_noise(text):

    # Step1: Make lowercase
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    # Step2: Remove whitespaces
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))

    # Step3 : Removing words inside brackets like "[OC]"
    text = text.apply(lambda x: re.sub(r"\[.*?\]", "", x))

    # Step4 : Removing everything from the data which is not alphanumeric.
    text = text.apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))

    # Step5 : Lemmatization
    lm=WordNetLemmatizer()
    text = text.apply(lambda x: lm.lemmatize(x))
    
    # Step6 : Removing Stopwords
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # Convert to string
    text = text.astype(str)
        
    return text
posts['title'] = remove_noise(posts['title'])
posts['title']
# Shuffling
posts = posts.sample(frac=1)
tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(posts['title'].values)
sequences = tokenizer.texts_to_sequences(posts['title'].values)
import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
totalNumWords = [len(one_comment) for one_comment in sequences]
plt.figure(figsize=(15, 10))
plt.hist(totalNumWords, bins=[i for i in range(1,70, 5)])
plt.show()
# Padding
MAX_LEN = 50
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)
padded_sequences
embedding_path = '/kaggle/input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
embed_size = 300
max_features = 30000
import numpy as np
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: 
        
        # Words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
embedding_matrix.shape
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(posts['flair'].values.reshape(-1, 1))
y_ohe
ohe.categories_
import pickle

# saving
with open('encoder.pickle', 'wb') as handle:
    pickle.dump(ohe, handle, protocol=pickle.HIGHEST_PROTOCOL)
file_path = "best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_accuracy", verbose = 1,
                              save_best_only = True, mode = "max")
early_stop = EarlyStopping(monitor = "val_accuracy", mode = "max", patience = 25)
def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (50,))
    x = Embedding(embedding_matrix.shape[0], embed_size, weights = [embedding_matrix], trainable = False)(inp)
    x1 = SpatialDropout1D(dr)(x)
    global history
    x_gru = Bidirectional(GRU(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x1)
    max_pool1_gru = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool3_gru = GlobalAveragePooling1D()(x3)
    max_pool3_gru = GlobalMaxPooling1D()(x3)
    
    x_lstm = Bidirectional(LSTM(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x1)
    max_pool1_lstm = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool3_lstm = GlobalAveragePooling1D()(x3)
    max_pool3_lstm = GlobalMaxPooling1D()(x3)
    
    
    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
                    avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(128,activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(100,activation='relu') (x))
    x = Dense(11, activation = "sigmoid")(x)
    
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(padded_sequences, y_ohe, batch_size = 128, epochs = 200, validation_split=0.3, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    import matplotlib.pyplot as plt
    print(history.history.keys())
   
    
    return model
model = build_model(lr = 1e-4, lr_d = 0, units = 128, dr = 0.5)
plt.figure(figsize=(15, 10))
plt.plot(history.history['accuracy'], label="acc")
plt.plot(history.history['val_accuracy'], label="val_Acc")
plt.grid()
plt.legend()
plt.show()
plt.figure(figsize=(15, 10))
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.grid()
plt.legend()
plt.show()
plot_model(model, to_file='./model.png')
def remove_noise_test(x):
  text = [" ".join(x.lower() for x in x[0].split())]
  text = [" ".join(x.strip() for x in text[0].split())]
  text = [" ".join(x for x in text[0].split() if x not in stop)]
  text = [re.sub(r"\[.*?\]", "", text[0])]
  text = [re.sub('[^a-zA-Z0-9\s]', '', text[0])]
  lm=WordNetLemmatizer()
  text = [lm.lemmatize(text[0])]
  text = tokenizer.texts_to_sequences(x)
  text = pad_sequences(text, maxlen=50, dtype='int32', value=0)
  return text
x = ["State Visit of Prime Minister Gandhi of India. President Reagan's Speech and Prime Minister Gandhi's Speech at Arrival Ceremony, South Lawn on July 29, 1982"]

text = remove_noise_test(x)

ans = model.predict(text, batch_size=1, verbose=2)

ohe.categories_[0][ans[0].argmax()]
x = ["Holy river Ganges self-cleared during lockdown. This was shot at Triveni Ghat, Rishikesh, Uttarakhand."]

text = remove_noise_test(x)

ans = model.predict(text, batch_size=1, verbose=2)

ohe.categories_[0][ans[0].argmax()]