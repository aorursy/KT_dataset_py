!pip install emoji
!pip install lazypredict
!pip install keras
!pip install sklearn
!conda install -c conda-forge hdbscan -y
##################################################
# Imports
##################################################

import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import emoji
from lazypredict.Supervised import LazyClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, plot_confusion_matrix
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from keras.initializers import Constant
from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import seaborn as sns
import hdbscan
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
from itertools import product
##################################################
# Params
##################################################

DATA_BASE_FOLDER = '/kaggle/input/emojify-challenge'


##################################################
# Utils
##################################################

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

# Use this function to evaluate your model
def accuracy(y_pred, y_true):
    '''
    input y_pred: ndarray of shape (N,)
    input y_true: ndarray of shape (N,)
    '''
    return (1.0 * (y_pred == y_true)).mean()
##################################################
# Load dataset
##################################################

df_train = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'train.csv'))
y_train = df_train['class']
df_validation = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'validation.csv'))
y_validation = df_validation['class']
emoji_dictionary = {
    '0': '\u2764\uFE0F',
    '1': ':baseball:',
    '2': ':smile:',
    '3': ':disappointed:',
    '4': ':fork_and_knife:'
}

# See some data examples
print('EXAMPLES:\n####################')
for idx in range(10):
    print(y_train[idx])
    print(f'{df_train["phrase"][idx]} -> {label_to_emoji(y_train[idx])}')
df_train.shape
# Load phrase representation
x_train = np.load(
    os.path.join(DATA_BASE_FOLDER, 
                 'train.npy')).reshape(len(df_train), -1)
x_validation = np.load(
    os.path.join(DATA_BASE_FOLDER, 
                 'validation.npy')).reshape(len(df_validation), -1)
print(f'Word embedding size: {x_train.shape[-1]}')

pd.DataFrame(x_train)
from scipy import stats
train_df = pd.DataFrame(x_train)

non_normal_features = list()

for i in range(0, train_df.shape[1]):
    shapiro_test = stats.shapiro(train_df.iloc[:, i]) # returns a tuple with two values: the test statistic and the pvalue
    if shapiro_test[1] < 0.05:
        non_normal_features.append(i)
        
print("Number of non normalized features: ", len(non_normal_features))
import random

random_features = random.sample(non_normal_features, 10)


fig, axs = plt.subplots(2, 5)
row = 0
col = 0
for _, i in enumerate(random_features):
    
    if col > 4:
        row = 1
        col = 0
        
    axs[row, col].hist(train_df.iloc[:, i], align='mid', alpha=0.6)
    
    col = col + 1
    
for ax in axs.flat:
    ax.set(xlabel='Values', ylabel='Count')
    
fig.savefig("random_non_normal_features.png")
f = plt.figure(figsize=(15,5))
f.add_subplot(1,2,1)
sns.countplot(y_train).set_title('Frequency of classes on response variable (Training Set)')
f.add_subplot(1,2,2)
sns.countplot(y_validation).set_title('Frequency of classes on response variable (Validation Set)')
def separate_words(x,y):
    x_new = x.ravel().reshape(-1,25)
    y_new = y.repeat(10).reset_index(drop=True)
    return (x_new, y_new)

x, y = separate_words(x_train, y_train)

words_index=np.where(~np.all(np.isclose(x, 0), axis=1))
clusterer = hdbscan.HDBSCAN(min_cluster_size=6)
clusterer.fit(x[words_index])
labels = clusterer.labels_
df = pd.DataFrame({'cluster':labels, 'emoji':y.iloc[words_index]})
df['words']=1
summary = df.groupby(['cluster', 'emoji']).agg('count')
pd.pivot_table(df, index='cluster', columns='emoji', aggfunc=lambda x: len(x), margins=False).iloc[1:].plot.barh(stacked=True, figsize=(15,10), title='Clustering of words in training set')
from collections import Counter
def separate_words(x,y):
    x_new = x.ravel().reshape(-1,25)
    y_new = y.repeat(10).reset_index(drop=True)
    return (x_new, y_new)

x, y = separate_words(x_train, y_train)

words_index=np.where(~np.all(np.isclose(x, 0), axis=1))
words_occurence = pd.DataFrame(Counter(map(tuple, x[words_index])).items(), columns=['Number of words', 'Occurrence']).groupby('Occurrence').count().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(data=words_occurence, x='Occurrence', y='Number of words').set_title('Words occurence distribution')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'\w+')

# Obtaining text
train_text = '. '.join(df_train['phrase'])
validation_text = '. '.join(df_validation['phrase'])

# Obtaining the tokens aka every syntatical element in the sentences (numbers, stopwords, words..)
train_words = tokenizer.tokenize(train_text)
validation_words = tokenizer.tokenize(validation_text)

# Removing commas and question mark and replacing them with a point in order to split the sentences
train_sentences = train_text.replace(",", ".").replace("?", ".").split(".")
validation_sentences = validation_text.replace(",", ".").replace("?", ".").split(".")

# Obtaining the length of each sentence for both datasets
train_sent_lenghs =[len(tokenizer.tokenize(sentence)) for sentence in train_sentences]
validation_sent_lenghs =[len(tokenizer.tokenize(sentence)) for sentence in validation_sentences]

# Storing the lengths previously obtained
train_sent_len = [i for i in train_sent_lenghs if i!=0]
validation_sent_len = [i for i in validation_sent_lenghs if i!=0]

# Plotting an histogram of the lengths-
plt.hist(train_sent_len, bins=range(min(train_sent_len), max(train_sent_len) + 1, 1),
              alpha=0.4, color="blue")

plt.hist(validation_sent_len, bins=range(min(validation_sent_len), max(validation_sent_len) + 1, 1), 
              alpha=0.4, color="red")

labels = ['Train',"Validation"]
plt.legend(labels)
plt.xlabel("Sentence Length")
plt.ylabel("Count")
plt.title("Comparing number of words per sentence distribution in Train and Validation sentences")

fig.savefig("train_validation_sentences_length.png")
from sklearn.linear_model import LogisticRegression

# Report the accuracy in the train and validation sets.
for c in (0.8, 0.5, 0.1,0.05, 0.01,0.001):
    log_class = LogisticRegression(fit_intercept=False, C=c)
    log_class.fit(x_train, y_train)
    y_pred = log_class.predict(x_validation)
    print(f'c = {c} accuracy={accuracy(y_pred, y_validation)}')
log_class = LogisticRegression(fit_intercept=False, C=0.5)
log_class.fit(x_train, y_train)
y_pred = log_class.predict(x_validation)
print(f'c={c} accuracy={accuracy(y_pred, y_validation)}')
print(classification_report(y_validation, y_pred, ))
plot_confusion_matrix(log_class, x_validation, y_validation,)
from numpy.linalg import norm

class SmartChebyshev():
    
    def __repr__(self): return 'smart_chebyshev'
    
    def _extract_words(self, tweet):
        all_words = tweet.reshape(-1,25)
        return all_words[np.where(~np.all(np.isclose(all_words, 0), axis=1))]
    
    def _distance(self, w1, w2):
        a = self._extract_words(w1)
        b = self._extract_words(w2)
        d = list()
        for x, y in product(a, b):
            d.append(norm(x-y))
        return max(d)
    
    def __call__(self, w1, w2):
        return self._distance(w1, w2)

sc = SmartChebyshev()

def test_knn(k=5, metric='euclidean', verbose=False):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(x_train, y_train)
    
    y_pred = knn.predict(x_validation)
    
    if verbose:
        print(f'Accuracy {accuracy(y_pred, y_validation)}')
        print(classification_report(y_validation, y_pred, ))
        return plot_confusion_matrix(knn, x_validation, y_validation)
    return accuracy(y_pred, y_validation)

res = list()

for k, m in product(range(1,11), ('euclidean', 'manhattan', 'chebyshev', sc)):
    res.append((k, m,  test_knn(k, metric=m)))
    
knn_tuning = pd.DataFrame(res, columns=['k', 'metric', 'accuracy'])
print(knn_tuning)
plt.figure(figsize=(10,5))
sns.lineplot(data=knn_tuning, x="k", y="accuracy", hue='metric')
test_knn(4,'euclidean', verbose=True)
x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy'))

# Reshaping feature datasets from 3D to 2D
x_validation_reshaped = x_validation.reshape((-1, 250))
x_test_reshaped = x_test.reshape((-1, 250))

# Keras requires one hot encoded outputs, so we prepare the corresponding format
categorical_y_train = to_categorical(y_train)
categorical_y_validation = to_categorical(y_validation)
def print_training_metrics(model):
    """
        Prints Accuracy and Validation Accuracy from the training procedure.
    """
    
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
from keras.utils.vis_utils import plot_model
naive_model = Sequential()
naive_model.add(Dense(64, activation = "relu", input_dim=250))
naive_model.add(Dropout(0.5))
naive_model.add(Dense(5, activation='softmax'))
naive_model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
naive_model.summary()
history = naive_model.fit(x_train, categorical_y_train, epochs=50)
y_pred = naive_model.predict(x_validation_reshaped)
y_pred = list(map(lambda x: np.argmax(x), y_pred))
accuracy(y_pred, y_validation)
# Defining the number of folds
K = 10

# Defining the loss and accuracy for each fold
K_fold_accuracy = list()
K_fold_loss = list()
K_fold_training_evolutions = [] # It will result in a list where each element has two nested lists, corresponding to the evolutions of both loss and accuracy during the training.
# We have to merge the train and validation dataset since sklearn's k-fold function will prepare the folds by itself.
features = np.concatenate((x_train, x_validation), axis = 0)
targets = np.concatenate((categorical_y_train, categorical_y_validation), axis = 0)

# Creating the Stratifield KFold object with the number of splits equal to K
skf = StratifiedKFold(n_splits = K, shuffle = True, random_state = 42) # 42 is our seed in order to provide reproducible results

n_fold = 1


checkpoint = ModelCheckpoint(os.path.join("./best_nn.h5"), 
    monitor='val_accuracy', verbose=1, 
    save_best_only=True, mode='max')
callbacks_list = [checkpoint]
    
for train, test in skf.split(features, targets.argmax(1)):
    
    naive_model = Sequential()
    naive_model.add(Dense(64, activation = "relu", input_dim=250))
    naive_model.add(Dropout(0.5))
    naive_model.add(Dense(5, activation='softmax'))
    naive_model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    
    print('------------------------------------------------------------------------')
    print(f'Training for fold {n_fold} ...')  
    history = naive_model.fit(features[train], targets[train],
              batch_size=32,
              epochs=50,
              verbose=1,
              callbacks = callbacks_list,
              validation_data=(features[test], targets[test]))
    
    # Generate generalization metrics
    scores = naive_model.evaluate(features[test], targets[test], verbose=0)
    print(f'Score for fold {n_fold}: {naive_model.metrics_names[0]} of {scores[0]}; {naive_model.metrics_names[1]} of {scores[1]*100}%')
    K_fold_accuracy.append(scores[1] * 100)
    K_fold_loss.append(scores[0])
    
    K_fold_training_evolutions.append(history)
    
    # Increase fold number
    n_fold = n_fold + 1
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(K_fold_accuracy)} (+- {np.std(K_fold_accuracy)})')
print(f'> Loss: {np.mean(K_fold_loss)}')
print('------------------------------------------------------------------------')
# Plots Train and Validation accuracy for each fold
def show_k_fold_accuracy(history_list, K):
    plt.figure(figsize=(15,8))
    plt.title('Train and Validation Accuracy by Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    palette = ["#fff100", "#ff8c00", "#e81123", "#ec008c", "#68217a", "#00188f", "#00bcf2", "#00b294", "#009e49", "#bad80a"]
    
    for i in range(0, K):
        plt.plot(history_list[i].history['accuracy'], label = 'Train Accuracy Fold ' + str(i+1), color = palette[i])
        plt.plot(history_list[i].history['val_accuracy'], label = 'Validation Accuracy Fold ' + str(i+1), color = palette[i], linestyle = "dashdot")

        plt.legend()
        
# Plots Train and Validation accuracy for each fold
def show_k_fold_loss(history_list, K):
    plt.figure(figsize=(15,8))
    plt.title('Train and Validation Loss by Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    palette = ["#fff100", "#ff8c00", "#e81123", "#ec008c", "#68217a", "#00188f", "#00bcf2", "#00b294", "#009e49", "#bad80a"]
    
    for i in range(0, K):
        plt.plot(history_list[i].history['loss'], label = 'Train Accuracy Loss ' + str(i+1), color = palette[i])
        plt.plot(history_list[i].history['val_loss'], label = 'Validation Accuracy Loss ' + str(i+1), color = palette[i], linestyle = "dashdot")

        plt.legend()
show_k_fold_accuracy(K_fold_training_evolutions, K)
show_k_fold_loss(K_fold_training_evolutions, K)
naive_model = load_model('./best_nn.h5')
naive_model.summary()
y_pred = naive_model.predict(x_validation_reshaped)
y_pred = list(map(lambda x: np.argmax(x), y_pred))
accuracy(y_pred, y_validation)
lambda_l = 0.02
# Defining the loss and accuracy for each fold
K_fold_accuracy = list()
K_fold_loss = list()
K_fold_training_evolutions = [] # It will result in a list where each element has two nested lists, corresponding to the evolutions of both loss and accuracy during the training.

# We have to merge the train and validation dataset since sklearn's k-fold function will prepare the folds by itself.
features = np.concatenate((x_train, x_validation), axis = 0)
targets = np.concatenate((categorical_y_train, categorical_y_validation), axis = 0)

# Creating the Stratifield KFold object with the number of splits equal to K
skf = StratifiedKFold(n_splits = K, shuffle = True, random_state = 42) # 42 is our seed in order to provide reproducible results

n_fold = 1

checkpoint = ModelCheckpoint(os.path.join("./best_nn_reg.h5"), 
    monitor='val_accuracy', verbose=1, 
    save_best_only=True, mode='max')
callbacks_list = [checkpoint]
    
for train, test in skf.split(features, targets.argmax(1)):
    
    naive_model_regularized = Sequential()
    naive_model_regularized.add(Dense(64, activation = "relu", input_dim=250, kernel_regularizer = l2(lambda_l)))
    naive_model_regularized.add(Dropout(0.5))
    naive_model_regularized.add(Dense(5, activation='softmax'))
    naive_model_regularized.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    naive_model_regularized.summary()


    print('------------------------------------------------------------------------')
    print(f'Training for fold {n_fold} ...')  
    history = naive_model_regularized.fit(features[train], targets[train],
              batch_size=32,
              epochs=50,
              verbose=1,
              callbacks = callbacks_list,
              validation_data=(features[test], targets[test]))
    
    # Generate generalization metrics
    scores = naive_model_regularized.evaluate(features[test], targets[test], verbose=0)
    print(f'Score for fold {n_fold}: {naive_model_regularized.metrics_names[0]} of {scores[0]}; {naive_model_regularized.metrics_names[1]} of {scores[1]*100}%')
    K_fold_accuracy.append(scores[1] * 100)
    K_fold_loss.append(scores[0])
    
    K_fold_training_evolutions.append(history)
    
    # Increase fold number
    n_fold = n_fold + 1
    

print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(K_fold_accuracy)} (+- {np.std(K_fold_accuracy)})')
print(f'> Loss: {np.mean(K_fold_loss)}')
print('------------------------------------------------------------------------')
show_k_fold_accuracy(K_fold_training_evolutions, K)
show_k_fold_loss(K_fold_training_evolutions, K)
naive_model_regularized = load_model('./best_nn_reg.h5')
naive_model_regularized.summary()
# We have to merge the train and validation dataset since sklearn's k-fold function will prepare the folds by itself.
features = np.concatenate((x_train, x_validation), axis = 0)
targets = np.concatenate((categorical_y_train, categorical_y_validation), axis = 0)

naive_model_regularized = Sequential()
naive_model_regularized.add(Dense(64, activation = "relu", input_dim=250, kernel_regularizer = l2(lambda_l)))
naive_model_regularized.add(Dropout(0.5))
naive_model_regularized.add(Dense(5, activation='softmax'))
naive_model_regularized.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
history = naive_model_regularized.fit(features, targets,
              batch_size=32,
              epochs=50,
              verbose=1)
features.shape
y_pred = naive_model_regularized.predict(x_validation_reshaped)
y_pred = list(map(lambda x: np.argmax(x), y_pred))
accuracy(y_pred, y_validation)
# NN submission

# Create submission
submission = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'sample_submission.csv'), dtype=str)
submission['class'] = submission['class'].astype("str")
x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy')).reshape(len(submission), -1)

y_test_pred = naive_model_regularized.predict(x_test)
y_test_pred = list(map(lambda x: np.argmax(x), y_test_pred))
y_test_pred = list(map(str, y_test_pred))

if y_test_pred is not None:
    submission['class'] = y_test_pred

submission['class'] = submission['class'].astype("str")
submission = submission[['Unnamed: 0', 'class']]
submission = submission.rename(columns={'Unnamed: 0': 'id'})

submission.to_csv('my_submission_nn.csv', index=False)
model_dnn = Sequential()
model_dnn.add(Dense(64, activation="relu", input_dim=250))
model_dnn.add(Dropout(0.5))
model_dnn.add(Dense(32, activation="relu"))
model_dnn.add(Dense(5, activation="softmax"))
model_dnn.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
model_dnn.summary()
plot_model(model_dnn, to_file='dnn_architecture.png', show_shapes=True, show_layer_names=True)
# Defining the loss and accuracy for each fold
K_fold_accuracy = list()
K_fold_loss = list()
K_fold_training_evolutions = [] # It will result in a list where each element has two nested lists, corresponding to the evolutions of both loss and accuracy during the training.

# Creating the Stratifield KFold object with the number of splits equal to K
skf = StratifiedKFold(n_splits = K, shuffle = True, random_state = 42) # 42 is our seed in order to provide reproducible results

n_fold = 1

checkpoint = ModelCheckpoint(os.path.join("./best_dnn.h5"), 
    monitor='val_accuracy', verbose=1, 
    save_best_only=True, mode='max')
callbacks_list = [checkpoint]

for train, test in skf.split(features, targets.argmax(1)):
    
    model_dnn = Sequential()
    model_dnn.add(Dense(64, activation="relu", input_dim=250))
    model_dnn.add(Dropout(0.5))
    model_dnn.add(Dense(32, activation="relu"))
    model_dnn.add(Dense(5, activation="softmax"))
    model_dnn.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

    print('------------------------------------------------------------------------')
    print(f'Training for fold {n_fold}')  
    history = model_dnn.fit(features[train], targets[train],
              batch_size=32,
              epochs=50,
              verbose=1,
              callbacks = callbacks_list,
              validation_data=(features[test], targets[test]))
    
    # Generate generalization metrics
    scores = model_dnn.evaluate(features[test], targets[test], verbose=0)
    print(f'Score for fold {n_fold}: {model_dnn.metrics_names[0]} of {scores[0]}; {model_dnn.metrics_names[1]} of {scores[1]*100}%')
    K_fold_accuracy.append(scores[1] * 100)
    K_fold_loss.append(scores[0])
    K_fold_training_evolutions.append(history)
    print(history.history.keys())   
    # Increase fold number
    n_fold = n_fold + 1

print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(K_fold_accuracy)} (+- {np.std(K_fold_accuracy)})')
print(f'> Loss: {np.mean(K_fold_loss)}')
print('------------------------------------------------------------------------')

show_k_fold_accuracy(K_fold_training_evolutions, K)
show_k_fold_loss(K_fold_training_evolutions, K)
model_dnn = load_model('./best_dnn.h5')
model_dnn.summary()

y_pred = model_dnn.predict(x_validation_reshaped)
y_pred = list(map(lambda x: np.argmax(x), y_pred))
accuracy(y_pred, y_validation)
model_dnn = Sequential()
model_dnn.add(Dense(64, activation="relu", input_dim=250))
model_dnn.add(Dropout(0.5))
model_dnn.add(Dense(32, activation="relu"))
model_dnn.add(Dense(5, activation="softmax"))
model_dnn.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
history = model_dnn.fit(features, targets,
              batch_size=32,
              epochs=50,
              verbose=1)
# DNN submission

# Create submission
submission = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'sample_submission.csv'))
x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy')).reshape(len(submission), -1)

y_test_pred = model_dnn.predict(x_test)
y_test_pred = list(map(lambda x: np.argmax(x), y_test_pred))

if y_test_pred is not None:
    submission['class'] = y_test_pred

submission = submission[['Unnamed: 0', 'class']]
submission = submission.rename(columns={'Unnamed: 0': 'id'})
submission.to_csv('my_submission_dnn.csv', index=False)
#### Loading the GloVe word embedding.

# Directory of the embedding file
filename = '../input/glove-global-vectors-for-word-representation/glove.twitter.27B.25d.txt'

# Dictionary where the key is the word and the value its internal index
index_to_words = {}
# Dictionary where the key is the index and the value its word 
words_to_index = {}
# Dictionary of word embeddings, where the key is the word and the value its embedding
word_embeddings = {}

i = 1
with open(filename, 'r') as f:
    words = set() # To avoid any kind of duplicates
    
    # Storing words and their embeddings
    for line in f:
        line = line.strip().split()
        current_word = line[0]
        words.add(current_word)
        word_embeddings[current_word] = np.array(line[1:], dtype=np.float64)

    # Performing the mapping between word and index
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i = i + 1
        
    
def convert_word_sentences_to_index_sentences(X, word_to_index, max_len):
    """
        This function takes a matrix of sentences, the mapping between word and indices and the maximum sentence length, and returns
        a matrix with the same shape as the one given in input but with the GloVe index in substition for each word.
        
        Input:
            - X: sentence matrix
            - word_to_index: dictionary of mappings between words and GloVe index
            - max_len: maximum sentence length
            
        Returns:
            - sentence_indices: matrix with the same shape as the one given as input, but with GloVe indices in substitution of the words.
    """
    n_rows = X.shape[0]
    sentence_indices = np.zeros((n_rows, max_len))
    
    # For each row, we perform the mapping by obtaining the words and, consecutively, their corresponding indices through mappings.
    for i in range(n_rows):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            sentence_indices[i, j] = word_to_index[w]
            j = j + 1
            
    return sentence_indices
# Preparing the embedding matrix where each row corresponds to the index assigned during the GloVe mapping, 
# and the 25 columns will contain its embedding representation.

number_of_words = len(words_to_index.keys())
embedded_matrix = np.zeros((number_of_words, 25))
    
print(embedded_matrix.shape)

for word, index in words_to_index.items():
    try:
        embedded_matrix[index, :] = word_embeddings[word]
    except:
        continue
features = np.concatenate((convert_word_sentences_to_index_sentences(df_train['phrase'].values, words_to_index, 10), convert_word_sentences_to_index_sentences(df_validation['phrase'].values, words_to_index, 10)), axis = 0)
targets = np.concatenate((to_categorical(y_train), to_categorical(y_validation)), axis = 0)
embedding_layer = Embedding(number_of_words, 25, embeddings_initializer=Constant(embedded_matrix), input_length=10 , trainable=False)
sequence_input = Input(shape=(10,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

convs = []
filter_sizes = [3,4,5]

for filter_size in filter_sizes:
    l_conv = Conv1D(filters=32, 
                    kernel_size=filter_size, 
                    activation='relu')(embedded_sequences)
    l_pool = GlobalMaxPooling1D()(l_conv)
    convs.append(l_pool)

conv_layers = concatenate(convs, axis=1)
x = Dropout(0.5)(conv_layers)  

predictions = Dense(5, activation='softmax')(x)
model_cnn = Model(sequence_input, predictions)
model_cnn.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

plot_model(model_cnn, to_file='cnn_architecture.png', show_shapes=True, show_layer_names=True)
# Defining the loss and accuracy for each fold
K_fold_accuracy = list()
K_fold_loss = list()
K_fold_training_evolutions = [] # It will result in a list where each element has two nested lists, corresponding to the evolutions of both loss and accuracy during the training.

# Creating the Stratifield KFold object with the number of splits equal to K
skf = StratifiedKFold(n_splits = K, shuffle = True, random_state = 42) # 42 is our seed in order to provide reproducible results

n_fold = 1

checkpoint = ModelCheckpoint(os.path.join("./best_cnn.h5"), 
    monitor='val_accuracy', verbose=1, 
    save_best_only=True, mode='max')
callbacks_list = [checkpoint]

for train, test in skf.split(features, targets.argmax(1)):
    embedding_layer = Embedding(number_of_words, 25, embeddings_initializer=Constant(embedded_matrix), input_length=10 , trainable=False)
    sequence_input = Input(shape=(10,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [3,4,5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=32, 
                        kernel_size=filter_size, 
                        activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)

    conv_layers = concatenate(convs, axis=1)
    x = Dropout(0.5)(conv_layers)  

    predictions = Dense(5, activation='softmax')(x)
    model_cnn = Model(sequence_input, predictions)
    model_cnn.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    print('------------------------------------------------------------------------')
    print(f'Training for fold {n_fold} ...')  
    history = model_cnn.fit(features[train], targets[train],
              batch_size=32,
              epochs=50,
              verbose=1,
              callbacks = callbacks_list,
              validation_data=(features[test], targets[test]))
    
    # Generate generalization metrics
    scores = model_cnn.evaluate(features[test], targets[test], verbose=0)
    print(f'Score for fold {n_fold}: {model_cnn.metrics_names[0]} of {scores[0]}; {model_cnn.metrics_names[1]} of {scores[1]*100}%')
    K_fold_accuracy.append(scores[1] * 100)
    K_fold_loss.append(scores[0])
    K_fold_training_evolutions.append(history)
    # Increase fold number
    n_fold = n_fold + 1
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(K_fold_accuracy)} (+- {np.std(K_fold_accuracy)})')
print(f'> Loss: {np.mean(K_fold_loss)}')
print('------------------------------------------------------------------------')
show_k_fold_accuracy(K_fold_training_evolutions, K)
show_k_fold_loss(K_fold_training_evolutions, K)
model_cnn = load_model('./best_cnn.h5')
model_cnn.summary()
embedding_layer = Embedding(number_of_words, 25, embeddings_initializer=Constant(embedded_matrix), input_length=10 , trainable=False)
sequence_input = Input(shape=(10,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

convs = []
filter_sizes = [3,4,5]

for filter_size in filter_sizes:
    l_conv = Conv1D(filters=32, 
                    kernel_size=filter_size, 
                    activation='relu')(embedded_sequences)
    l_pool = GlobalMaxPooling1D()(l_conv)
    convs.append(l_pool)

conv_layers = concatenate(convs, axis=1)
x = Dropout(0.5)(conv_layers)  

predictions = Dense(5, activation='softmax')(x)
model_cnn = Model(sequence_input, predictions)
model_cnn.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

history = model_cnn.fit(features, targets,
          batch_size=32,
          epochs=50,
          verbose=1)
X = convert_word_sentences_to_index_sentences(df_validation['phrase'].values, words_to_index, 10)
y = to_categorical(y_validation)

y_pred = model_cnn.predict(X)

loss, acc = model_cnn.evaluate(X, y)
# CNN submission

submission = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'sample_submission.csv'))

submission['phrase'] = submission['phrase'].str.replace("\t", "")
X = convert_word_sentences_to_index_sentences(submission['phrase'].values, words_to_index, 10)

y_pred = model_cnn.predict(X)
y_pred = list(map(lambda x: np.argmax(x), y_pred))

if y_pred is not None:
    submission['class'] = y_pred

submission = submission[['Unnamed: 0', 'class']]
submission = submission.rename(columns={'Unnamed: 0': 'id'})
submission.to_csv('my_submission_cnn.csv', index=False)
model_lstm = Sequential() 
model_lstm.add(Embedding(number_of_words, 25,
                           embeddings_initializer=Constant(embedded_matrix)))
model_lstm.add(Bidirectional(LSTM(128)))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(units=5, activation='softmax'))
model_lstm.add(Activation('softmax'))
model_lstm.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

plot_model(model_lstm, to_file='lstm_architecture.png', show_shapes=True, show_layer_names=True)
# Defining the loss and accuracy for each fold
K_fold_accuracy = list()
K_fold_loss = list()
K_fold_training_evolutions = [] # It will result in a list where each element has two nested lists, corresponding to the evolutions of both loss and accuracy during the training.

# Creating the Stratifield KFold object with the number of splits equal to K
skf = StratifiedKFold(n_splits = K, shuffle = True, random_state = 42) # 42 is our seed in order to provide reproducible results

n_fold = 1

checkpoint = ModelCheckpoint(os.path.join("./best_lstm.h5"), 
    monitor='val_accuracy', verbose=1, 
    save_best_only=True, mode='max')
callbacks_list = [checkpoint]

for train, test in skf.split(features, targets.argmax(1)):

    model_lstm = Sequential() 
    model_lstm.add(Embedding(number_of_words, 25,
                               embeddings_initializer=Constant(embedded_matrix)))
    model_lstm.add(Bidirectional(LSTM(128)))
    model_lstm.add(Dropout(0.5))
    model_lstm.add(Dense(units=5, activation='softmax'))
    model_lstm.add(Activation('softmax'))
    model_lstm.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    
    
    print('------------------------------------------------------------------------')
    print(f'Training for fold {n_fold} ...')  
    history = model_lstm.fit(features[train], targets[train],
              batch_size=32,
              epochs=50,
              verbose=2,
              callbacks = callbacks_list,
              validation_data=(features[test], targets[test]))
    
    # Generate generalization metrics
    scores = model_lstm.evaluate(features[test], targets[test], verbose=0)
    print(f'Score for fold {n_fold}: {model_lstm.metrics_names[0]} of {scores[0]}; {model_lstm.metrics_names[1]} of {scores[1]*100}%')
    K_fold_accuracy.append(scores[1] * 100)
    K_fold_loss.append(scores[0])
    K_fold_training_evolutions.append(history)
    # Increase fold number
    n_fold = n_fold + 1
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(K_fold_accuracy)} (+- {np.std(K_fold_accuracy)})')
print(f'> Loss: {np.mean(K_fold_loss)}')
print('------------------------------------------------------------------------')

show_k_fold_accuracy(K_fold_training_evolutions, K)
show_k_fold_loss(K_fold_training_evolutions, K)
model_lstm = load_model('./best_lstm.h5')
model_lstm.summary()
model_lstm = Sequential() 
model_lstm.add(Embedding(number_of_words, 25,
                           embeddings_initializer=Constant(embedded_matrix)))
model_lstm.add(Bidirectional(LSTM(128)))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(units=5, activation='softmax'))
model_lstm.add(Activation('softmax'))
model_lstm.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

history = model_lstm.fit(features, targets,
          batch_size=32,
          epochs=50,
          verbose=2)
X = convert_word_sentences_to_index_sentences(df_validation['phrase'].values, words_to_index, 10)
y = to_categorical(y_validation)

y_pred = model_lstm.predict(X)

loss, acc = model_lstm.evaluate(X, y)
# LSTM submission

submission = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'sample_submission.csv'))
x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy')).reshape(len(submission), -1)

submission['phrase'] = submission['phrase'].str.replace("\t", "")

X = convert_word_sentences_to_index_sentences(submission['phrase'].values, words_to_index, 10)

y_pred = model_lstm.predict(X)
y_pred = list(map(lambda x: np.argmax(x), y_pred))

if y_pred is not None:
    submission['class'] = y_pred

submission = submission[['Unnamed: 0', 'class']]
submission = submission.rename(columns={'Unnamed: 0': 'id'})
submission.to_csv('my_submission_lstm.csv', index=False)