# Dataframes
import pandas as pd

# SK-Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#NLTK
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# String manipulation
import re
import string

# Random
import random

# Keras
from keras_preprocessing.text import Tokenizer
from keras import Sequential, initializers, regularizers, layers, utils, Input

# Plotting
import matplotlib.pyplot as plt

# Numpy
import numpy as np

# Itertools
import itertools

# Function to remove html links & mentions
def remove_html_mentions(text):
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    return text

# Function to remove numbers
def remove_numbers(text):
    text = ''.join([''.join([i for i in word if not i.isdigit()]) for word in text])
    return text

# Function to remove punctuation
def remove_punctuation(text):
    text = ''.join([symbol for symbol in text if symbol not in string.punctuation])
    return text

tokenizer = RegexpTokenizer('\s+', gaps=True)
# Function to tokenize text
def tokenize_text(text):
    tokenized_text = tokenizer.tokenize(text)
    return tokenized_text

# Function to remove stopwords
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

lemmatizer = WordNetLemmatizer()
# Function to lemmatize words:
def lemmatize_words(text):
    lem_text = [lemmatizer.lemmatize(word) for word in text]
    return lem_text

stemmer = PorterStemmer()
# Function to stem words
def stem_words(text):
    stem_text = ' '.join([stemmer.stem(word) for word in text])
    return stem_text
def clean(data):
    data = data.apply(lambda x: remove_html_mentions(x))
    data = data.apply(lambda x: remove_numbers(x))
    data = data.apply(lambda x: remove_punctuation(x))
    data = data.apply(lambda x: tokenize_text(x.lower()))
    data = data.apply(lambda x: remove_stopwords(x))
    data = data.apply(lambda x: lemmatize_words(x))
    data = data.apply(lambda x: stem_words(x))
    return data
def train_test_validate(x, y, t_size=.25, validation_size=.25):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=1)

    val_size = round((1-t_size)*validation_size,2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=1)

    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    validate = pd.concat([x_val, y_val], axis=1)

    return train, test, validate
f = '../input/sentiment140/training.1600000.processed.noemoticon.csv'

# Total number of data points
num_lines = sum(1 for _ in open(f, encoding='ISO-8859-1'))

# Number of data points to use
size = int(num_lines / 50)
skip_idx = random.sample(range(1, num_lines), num_lines - size)
tweet_data = pd.read_csv(f, skiprows=skip_idx, encoding='ISO-8859-1')
# Set column names
tweet_data.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
# Reset target columns
tweet_data.loc[tweet_data['target'] == 0, 'target'] = 0
tweet_data.loc[tweet_data['target'] == 4, 'target'] = 1
target, text = tweet_data.iloc[:,0], tweet_data.iloc[:,5]
# Clean tweets
text = clean(text)
# Create train, validation and test data
train, test, validate = train_test_validate(text, target)
# Create tokenizer
num_words_keep = 1000
tokenizer = Tokenizer(num_words=num_words_keep,filters='',lower=False,split=' ',
                      char_level=False, oov_token=None)
# Fit tokenizer on training data
x_train = train.iloc[:,0]

tokenizer.fit_on_texts(texts=x_train)
modes = ['binary', 'count', 'tfidf', 'freq']
# Training data
y_train = train.iloc[:,1]

x_train = tokenizer.texts_to_matrix(x_train, mode=modes[1])
y_train = utils.to_categorical(y_train, num_classes=2)
# Validation data
x_validate = validate.iloc[:,0]
x_validate = tokenizer.texts_to_matrix(x_validate, mode=modes[1])

y_validate = validate.iloc[:,1]
y_validate = utils.to_categorical(y_validate, num_classes=2)
# Test data
x_test = test.iloc[:,0]
x_test = tokenizer.texts_to_matrix(x_test, mode=modes[1])

y_test = test.iloc[:,1]
# Class Names for labels 0:Negative, 1:Positive
class_names = ['Negative', 'Positive']
# Implement model
model = Sequential()

activation_functions = ['relu','sigmoid','tanh']

# Kernel/weight initialiser
initialiser = initializers.GlorotNormal(seed=1) # Mitigating risk of vanishing/exploding gradients

# Kernel/weight regulariser
reg_constant1 = 0.01
l2_regulariser = regularizers.l2(l=reg_constant1)

# Dropout 20% of input variables at random each pass
dropout = layers.Dropout(.2, input_shape=(num_words_keep,))

layer1 = layers.Dense(
    units = num_words_keep/2,
    activation= activation_functions[0],
    use_bias=True,
    kernel_initializer=initialiser,
    bias_initializer='zeros',
    kernel_regularizer=l2_regulariser
)

layer2 = layers.Dense(
    units = 2,
    activation= activation_functions[1],
    use_bias=True,
    kernel_initializer=initialiser,
    bias_initializer='zeros'
)

model.add(Input(shape=(num_words_keep,)))
model.add(dropout)
model.add(layer1)
model.add(layer2)

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Train model
history = model.fit(x=x_train,
                    y=y_train,
                    batch_size=512,
                    epochs=10,
                    verbose=2, #one line per epoch
                    validation_data=(x_validate,y_validate),
                    shuffle=True,
                    validation_freq=1)
# Plot accuracy
plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
# Plot loss
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.title('Training & Validation Loss')
plt.ylabel('Loss')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
plt.close()
# Function to plot confusion matrix
# Credit: (https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis/notebook)
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
# Make new predictions:
y_pred = np.argmax(model.predict(x_test), axis=-1)
# Plot confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,8))
plot_confusion_matrix(cm=cnf_matrix, classes=class_names)
plt.show()
plt.close()
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))