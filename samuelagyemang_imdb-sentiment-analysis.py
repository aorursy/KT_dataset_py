# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#imports
import os
import string
import numpy as np # linear algebra
import pandas as pd
import random
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional,Conv2D, Conv1D, GlobalMaxPooling1D, Dropout, ReLU, \
                                    Reshape, MaxPooling2D, Concatenate, TimeDistributed,Flatten
from keras import regularizers
from keras.utils.np_utils import to_categorical
from tensorflow.keras.initializers import Constant
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import CuDNNLSTM, CuDNNGRU
#Read reviews from csv file
df = pd.read_csv("../input/imdb-review-dataset/imdb_master.csv", encoding='ISO-8859-1')
#Process all reviews
def process_reviews(reviews):
    p_reviews=[]
    for review in reviews:
        #remove html tags
        soup = BeautifulSoup(review)
        review = soup.get_text()
        #tokenize review: separate each review them into words
        #TODO try sentence tokenizer
        tokens = word_tokenize(review)
        #convert words to lower case
        tokens = [t.lower() for t in tokens]
        #remove punctuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [t.translate(table) for t in tokens]
        #remove non alphabetic tokens
        words = [word for word in stripped if word.isalpha()]
        #filter stopwords
        stop_words = set(stopwords.words('english'))
        words = [w for w in words  if not w in stop_words]
        words = [wnl.lemmatize(w) for w in words]
        p_reviews.append(words)
        
    return p_reviews
def blstm_model(num_words, e_size, e_matrix, max_length, activation):
    model = Sequential()
    embedding_layer = Embedding(num_words,
                                e_size,
                                embeddings_initializer = Constant(e_matrix), 
                                input_length = max_length,
                                trainable=True)
    model.add(embedding_layer)
    model.add(Bidirectional(CuDNNLSTM(100, dropout = 0.2, recurrent_dropout= 0.2)))
    model.add(Dense(2, activation=activation))
    
    return model
# CNN model implemented by Kim Yoon
# Paper: Convolutional Neural Networks for Sentence Classification
# url: https://www.aclweb.org/anthology/D14-1181/

# Hyperparameters
def yoon_cnn(num_words, embedding_size, embedding_matrix, input_length):
    filter_sizes = [3, 4, 5]  # defined convs regions
    num_filters = 100  # num_filters per conv region
    drop = 0.5

    embedding_layer = Embedding(num_words,
                                embedding_size,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=input_length,
                                trainable=True)

    inputs = Input(shape=(input_length,), dtype='int32')
    embedding = embedding_layer(inputs)
    reshape = Reshape((input_length, embedding_size, 1))(embedding)

    conv_0 = Conv2D(num_filters, (filter_sizes[0], embedding_size), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], embedding_size), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], embedding_size), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)

    maxpool_0 = MaxPooling2D((input_length - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0)
    maxpool_1 = MaxPooling2D((input_length - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_1)
    maxpool_2 = MaxPooling2D((input_length - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_2)

    merged_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(merged_tensor)
    dropout1 = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(dropout1)

    # this creates a model that includes
    model = Model(inputs, output)
    return model
def get_metrics(test_labels, predictions):
    print('Confusion Matrix')
    print(confusion_matrix(test_labels.argmax(axis=1), predictions))
    print('')
    print('Classification Report')
    print(classification_report(test_labels.argmax(axis=1), predictions))
def cnn_model(num_words, embedding_size, embedding_matrix, input_length):
    model = Sequential()

    model.add(Embedding(num_words,
                        embedding_size,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=input_length,
                        trainable=True))
    model.add(Dropout(0.2))
    model.add(Conv1D(250,3,padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(250))
    model.add(Dropout(0.2))
    model.add(ReLU())
    model.add(Dense(2, activation='softmax'))

    return model
def create_callbacks(best_model_path, monitor, mode, patience):
    es = EarlyStopping(monitor=monitor, mode=mode, verbose=1, patience=patience)
    mc = ModelCheckpoint(best_model_path, monitor=monitor, mode=mode, verbose=1, save_best_only=True)
    callbacks = [es, mc]
    return callbacks

def plot_graph(title, train_hist, val_hist, x_label, y_label, legend, loc):
    plt.plot(train_hist)
    plt.plot(val_hist)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend, loc=loc)
#     plt.savefig(path)
    plt.show()
    plt.clf()

def decode_one_hot_labels(test_labels):
    return np.argmax(test_labels, axis=1)

def get_metrics(test_labels, predictions):
    print('Confusion Matrix')
    print(confusion_matrix(test_labels.argmax(axis=1), predictions))
    print('')
    print('Classification Report')
    print(classification_report(test_labels.argmax(axis=1), predictions))


df.head()
#Get all reviews and labels
reviews = df.loc[((df['type'] == 'train') | (df['type'] == 'test')) & ((df['label'] == 'neg') | (df['label'] == 'pos'))]

print(reviews.shape)
print(reviews['label'].unique())
#Change values of labels
#neg = 0 pos = 1
reviews.loc[reviews['label']=='neg', 'label'] = 0
reviews.loc[reviews['label']=='pos', 'label'] = 1
print(reviews['type'].unique())
print(reviews['label'].unique())
reviews.head()
#Split review and label dataframe
review_data = reviews['review']
review_labels = reviews['label']

review_data = review_data.tolist()
review_labels = review_labels.tolist()
#Largest number of words in the reviews
max_length = max(len(r.split()) for r in review_data)
print(max_length)
print(review_data[0])
#Initialize lemmatizer
wnl = WordNetLemmatizer()
#Example
s_review = wnl.lemmatize('movies')
print(s_review)
#Clean the data
clean_reviews = process_reviews(review_data)
print(type(clean_reviews))
#Train the Word2Vec model with a embedding size=100
embedding_size = 100

w2v_model = gensim.models.Word2Vec(sentences = clean_reviews, size = embedding_size, window=5, workers=10, min_count=1, iter=10)
words = list(w2v_model.wv.vocab)
print(len(words))
print(w2v_model.wv.most_similar('love'))
print('')
print(w2v_model.wv.most_similar_cosmul(positive = ['woman','king'], negative = ['man']))
w2v_model.wv.doesnt_match('chinese british coffee spanish'.split())
#Save embeddings to file
file = "w2v_embeddings_imdb.txt"
w2v_model.wv.save_word2vec_format(file, binary=False)
#Read embeddings file
embeddings_file = open(os.path.join('','w2v_embeddings_imdb.txt'), encoding='utf-8')

dictionary = {}

#Create a dictionary of each word and its learned vectors
for line in embeddings_file:
    values = line.split()
    word = values[0]
    vectors = np.asarray(values[1:])
    dictionary[word] = vectors
    
embeddings_file.close()
dictionary['car']
#Instantiate a Tokenizer: it assigns each word in the corpus a integer values
tokenizer = Tokenizer()
#This method creates the vocabulary index based on word frequency
tokenizer.fit_on_texts(clean_reviews)
#It takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary.
sequences = tokenizer.texts_to_sequences(clean_reviews)
print(sequences[1])
print('')
print(sequences[0:3])
#Pad sequnces to the same length
padded_reviews = pad_sequences(sequences, maxlen = max_length, padding='post')
#Convert labels to numpy array
review_labels = np.asarray(review_labels)
word_index = tokenizer.word_index
print('Unique tokens: '+str(len(word_index)))
print('Shape of reviews:', padded_reviews.shape)
print('Shape of review labels:',review_labels.shape)
print(type(word_index))
num_words = len(word_index)+1

#(180305, 100)
embedding_matrix = np.zeros((num_words, embedding_size))

#word_index(word, number of coressponding word)
for word, num in word_index.items():
    #if int value of word is > the token dictionary size, ignore
    if num > num_words:
        continue
    #Get features vectors for each word in token dictionary from our trained embedding layer i.e w2v
    #Store in the embedding matrix(number coressponding to word, feature vectors)
    embedding_vector = dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[num] = embedding_vector
    else:
        embedding_matrix[num] = np.random.randn(embedding_size)
# detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# # instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # instantiating the model in the strategy scope creates the model on the TPU

    #     model = tf.keras.Sequential( … ) # define your model normally
    #     model.compile( … )
#Shuffle out data
indices = np.arange(padded_reviews.shape[0])
np.random.shuffle(indices)

padded_reviews = padded_reviews[indices]
review_labels = review_labels[indices]
# print(len(padded_reviews))

#Split our data in training and test data 80:20
X_train, X_test, y_train, y_test = train_test_split(padded_reviews, review_labels, test_size=0.2)

#OneHot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# Initialize mode
best_model_path = "semtiment_model.h5"
callbacks = create_callbacks(best_model_path, 'val_loss', 'min', 2)
        
model = yoon_cnn(num_words, embedding_size, embedding_matrix, max_length)
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
BATCH_SIZE = 64
# x_train[0]
# Training a BiLSTM takes forever
epochs = 200
history = model.fit(X_train, 
                    y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=epochs, 
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1)
#Test model
score,acc = model.evaluate(X_test, y_test, verbose = 2, batch_size = BATCH_SIZE)
print("Test score: %.4f" % (score))
print("Model Accuracy: %.4f" % (acc))

preds = model.predict(X_test, verbose=2)
preds = np.argmax(preds, axis=1)
#Show metrics
get_metrics(y_test,preds)

accuracy = accuracy_score(y_test.argmax(axis=1),preds)
print('Accuracy: ', str(accuracy))

precision = precision_score(y_test.argmax(axis=1),preds,average='binary')
print('Precision: ', str(precision))

recall = recall_score(y_test.argmax(axis=1),preds,average='binary')
print('Recall: ', str(recall))

f1 = f1_score(y_test.argmax(axis=1), preds, average='binary')
print('F1 score: ', str(f1))
plot_graph('Accuracy', history.history['accuracy'], history.history['val_accuracy'], 
           'epochs', 'Acc', ['train', 'val'], 'upper left')

plot_graph('Loss', history.history['loss'], history.history['val_loss'], 
           'epochs', 'Loss', ['train', 'val'], 'upper left')
#Clean a reviews

def clean_reviews(user_reviews):

  usr_reviews = []

  for user_review in user_reviews:
    #remove html tags
    soup = BeautifulSoup(user_review)
    user_review = soup.get_text()
    #tokenize review: separate each review them into words
    #TODO try sentence tokenizer
    tokens = word_tokenize(user_review)
    #convert words to lower case
    tokens = [t.lower() for t in tokens]
    #remove punctuations
    table = str.maketrans('', '', string.punctuation)
    stripped = [t.translate(table) for t in tokens]
    #remove non alphabetic tokens
    words = [word for word in stripped if word.isalpha()]
    #filter stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words  if not w in stop_words]
    words = [wnl.lemmatize(w) for w in words]

    usr_reviews.append(words)

  return usr_reviews

#Get max length
def get_max_length(data):
  list_len = [len(i) for i in data]
  max_length = max(list_len)
  print(max_length)

def tokenize_reviews(data, tokenizer):
  # user_tokenizer = Tokenizer()
  # user_tokenizer.fit_on_texts(data)
  user_sequences = tokenizer.texts_to_sequences(data)

  return user_sequences

def pad_reviews(seq, max_l):
    padded = pad_sequences(seq, maxlen = max_l, padding='post')
    return padded

def display_results(reviews,preds):
  pos = 0
  n_preds = preds.argmax(axis=1)
  for i in range(0, len(reviews)):
    print(reviews[i])
    if n_preds[i] == 0:
        print('*negative*')
        print(' Confidence: '+ str(round((preds[i].max() * 100),2))+'%')
    else:
      print('*positive*')
      print(' Confidence: '+ str(round((preds[i].max() * 100),2))+'%')
      pos += 1

    print('------------------------------------')

  pos_percent = (pos/len(reviews))*100
  print('#Total reviews: ', len(reviews))
  print('#Positive reviews: ', pos)
  print("#Total positives:", round(pos_percent,2))
#Predictions
user_reviews = ["My family and I normally do not watch local movies for the simple reason that they are poorly made, they lack the depth, and just not worth our time.<br /><br />The trailer of \"Nasaan ka man\" caught my attention, my daughter in law's and daughter's so we took time out to watch it this afternoon. The movie exceeded our expectations. The cinematography was very good, the story beautiful and the acting awesome. Jericho Rosales was really very good, so's Claudine Barretto. The fact that I despised Diether Ocampo proves he was effective at his role. I have never been this touched, moved and affected by a local movie before. Imagine a cynic like me dabbing my eyes at the end of the movie? Congratulations to Star Cinema!! Way to go, Jericho and Claudine!!",
                "Believe it or not, this was at one time the worst movie I had ever seen. Since that time, I have seen many more movies that are worse (how is it possible??) Therefore, to be fair, I had to give this movie a 2 out of 10. But it was a tough call.",
                "This was the worst movie in my entire life.",
                "I will recommend this movie to the entire universe.",
                "Even aside from its value as pure entertainment, this movie can serve as a primer to young adults about the tensions in the Middle East.",
                "Derivative, uneven, clumsy and absurdly sexist.",
                "I've seen it twice and it's even better the second time.",
                "This was a very good movie. I really enjoyed it.",
                "Horrible waste of time. I do not recommend this movie to anyone",
                "Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he's better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutcher's ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in.",
                "At the bottom end of the apocalypse movie scale is this piece of pish called 'The Final Executioner'.. at least where I come from. A bloke is trained by an ex-cop to seek vengeance on those that killed his woman and friends in cold blood.. and that's about it. Lots of fake explosions and repetitive shootings ensue. Has one of the weirdest array of costumes I've seen in a film for a while, and a massive fortress which is apparently only run by 7 people. GREAT job on the dubbing too guys(!) Best moment: when our hero loses a swordfight and is about to be skewered through the neck, he just gets out his gun and BANG! Why not do that earlier? It's a mystery. As is why anyone would want to sit through this in the first place. I'm still puzzling over that one myself now.. 2/10",
                "A boy and a girl is chased by a local warrior because the boy killed (by accident) the warriors father (or whoever he was). And they travel through the nature of Africa's most ruff areas.<br /><br />The acting in this movie isn't that good (except for that elephant kid). But it's a very good adventure and it's not very censored, there is some blood, flesh and nudity (which lighten up the movie a bit).<br /><br />I give this movie a 7 because of it's picture of the African nature and it's action.",
                "Hell yeah!!! For someone who always sleeps while watching a movie, I can I liked it. Though I didnt understand language and relied on the subtitles. I learnt a new culture.",
                "I would watch this movie again."] 


#Clean reviews
processed_user_reviews = clean_reviews(user_reviews)
# print(processed_user_reviews)

#Tokenize data
user_sequences = tokenize_reviews(processed_user_reviews, tokenizer)
# print(user_sequences)

#Pad the reviews
padded_user_reviews = pad_reviews(user_sequences, max_length)
# print(padded_user_reviews)

#Predict sentiment
preds = model.predict(padded_user_reviews)
display_results(user_reviews, preds)