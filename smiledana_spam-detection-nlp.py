stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
             "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", 
             "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", 
             "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", 
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", 
             "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", 
             "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", 
             "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", 
             "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", 
             "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
             "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", 
             "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where",
             "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", 
             "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
import os
import csv
import tensorflow as tf
import numpy as np
labels = []
messages = []
with open("../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv", 'r') as csvfile:
  csvreader = csv.reader(csvfile, delimiter=',')
  next(csvreader)
  for row in csvreader:
    labels.append(row[0])
    sentence = row[1]
    for word in stopwords:
      token = " " + word + " "
      sentence = sentence.replace(token, " ")
    
    messages.append(sentence)
print("Total data number: ",len(messages))
print(messages[0])


training_ratio = 0.8
train_size = int(training_ratio * len(messages))
train_messages = messages[:train_size]
train_labels = labels[:train_size]
valid_messages = messages[train_size:]
valid_labels = labels[train_size:]
print("training data: ", len(train_messages))
print("validation data: ",len(valid_messages))
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words = 1000, oov_token='<oov>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True)
tokenizer.fit_on_texts(train_messages)
word_index = tokenizer.word_index 
print(word_index)
train_sequences = tokenizer.texts_to_sequences(train_messages)
train_padded = pad_sequences(train_sequences,  maxlen=85, padding='post', truncating='post')#maxlen=85 can be known after run without maxlen
valid_sequences = tokenizer.texts_to_sequences(valid_messages)
valid_padded = pad_sequences(valid_sequences,maxlen=85, padding='post', truncating='post')
print(train_sequences[0])
print(train_padded.shape)
print(valid_padded.shape)
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
train_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
valid_label_seq = np.array(label_tokenizer.texts_to_sequences(valid_labels))
print(train_label_seq.shape)
print(valid_label_seq.shape)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=85),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 30
history = model.fit(train_padded, train_label_seq, epochs=num_epochs, 
                    batch_size=128,
                    validation_data=(valid_padded, valid_label_seq)) 
results = model.evaluate(valid_padded, valid_label_seq)
print(results)
import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
import pandas as pd

data = pd.read_csv("../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv",delimiter=',')
data.info()
# convert categorical variable into dummy/indicator variables
data['Category'] = pd.get_dummies(data['Category'], drop_first=True)
import nltk as nlp
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

sentences_final = []
for sentences in data["Message"]:
    sentences = re.sub("[^a-zA-Z]"," ",sentences)
    sentences = sentences.lower()   # buyuk harftan kucuk harfe cevirme
    sentences = nlp.word_tokenize(sentences)
    lemma = nlp.WordNetLemmatizer()
    Stopwords = stopwords.words("english")
    for word in sentences:
        if not word in Stopwords:
            lemma.lemmatize(word)
    sentences = " ".join(sentences)   
    sentences_final.append(sentences)
from sklearn.feature_extraction.text import CountVectorizer 
max_features = 5000 #5000 most common words
vect = CountVectorizer(max_features = max_features, stop_words = "english")
sparce_matrix = vect.fit_transform(sentences_final).toarray() # fit CountVectorizer and converting features/target into numeric vector
#print("the most using {} words: {}".format(max_features,vect.get_feature_names()))

# split dataset into train/test
y = data['Category']  
x = sparce_matrix
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 52)
# loading all classifiers
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
svc = SVC(kernel = 'linear')
mnb = MultinomialNB(alpha =0.2)
gnb  = GaussianNB()
lr = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=100,random_state=52)
abc = AdaBoostClassifier(n_estimators =100,random_state=52)
knn = KNeighborsClassifier(n_neighbors = 2)

# define a dictionary of classifier
classifier={'SVM': svc , 'MultinomialNB': mnb,'GaussianNB': gnb,'logistic': lr,'RandomForest': rfc,'Adaboost': abc, 'KNN':knn}
accuracy = []
for label, model in classifier.items():
    model.fit(X_train,y_train)
    accuracy.append((label,[model.score(X_test,y_test)]))
accuracy_df = pd.DataFrame(accuracy)
accuracy_df.columns = ["Classifier","Accuracy"]
accuracy_df