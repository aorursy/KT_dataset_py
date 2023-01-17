import numpy as np 
import pandas as pd
df1 = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json",lines = True)
df2 = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines = True)

df = pd.concat([df1,df2],sort = False)
df.tail()


%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x = "is_sarcastic", data = df)
plt.title("Data Distribution")
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 

stop_words =set(stopwords.words("english"))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import PorterStemmer 

def preprocess(text):
  word_list = []
  tok = tokenizer.tokenize(text)
  for word in tok:
    if word not in stop_words:
      word_list.append(stemmer.stem(word))
  return " ".join(word_list)
x_data = df["headline"].apply(preprocess)
x_data.tail()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, df['is_sarcastic'].values, test_size=0.10, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(X_train)
x_test_tfidf = vectorizer.transform(X_test)

df_idf = pd.DataFrame(vectorizer.idf_, index=vectorizer.get_feature_names(),columns=["idf_weights"])
 

df_idf.sort_values(by=['idf_weights']).tail()


from sklearn.naive_bayes import MultinomialNB

nb=MultinomialNB()

nb.fit(x_train_tfidf, y_train)
print(f"Training Score : {nb.score(x_train_tfidf, y_train)}")
print(f"Test Score : {nb.score(x_test_tfidf, y_test)}")
from  sklearn.metrics import confusion_matrix,classification_report
print("Confusion Matrix:")
print(confusion_matrix(y_test,nb.predict(x_test_tfidf)))
print("Summary")
print(classification_report(y_test,nb.predict(x_test_tfidf)))
sns.heatmap(confusion_matrix(y_test,nb.predict(x_test_tfidf)),annot=True,cmap='rainbow')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train_tfidf,y_train)
print(f"Training Score : {rf.score(x_train_tfidf, y_train)}")
print(f"Test Score : {rf.score(x_test_tfidf, y_test)}")
print("Confusion Matrix")
print(confusion_matrix(y_test,rf.predict(x_test_tfidf)))
print("Summary")
print(classification_report(y_test,rf.predict(x_test_tfidf)))
sns.heatmap(confusion_matrix(y_test,rf.predict(x_test_tfidf)),annot=True,cmap='rainbow')
from keras.models import Model

from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten, LSTM, Input, Embedding
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import BatchNormalization , Activation
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df[['headline', 'is_sarcastic']], test_size=0.1)  

training_sentences = list(train_data['headline'])
training_labels = list(train_data['is_sarcastic'])

testing_sentences = list(test_data['headline'])
testing_labels = list(test_data['is_sarcastic'])
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
vocab_size = 10000 
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)


vocab_size = 10000 
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"
#Building LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 10
history = model.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))
#Evaluating Accuracy and Loss of the LSTM model
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
plt.show()
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 10
history = model.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))
#Evaluating Accuracy and Loss of the CNN model
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
plt.show()