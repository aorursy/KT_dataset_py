import pandas as pd
from nltk.corpus import stopwords
import re
import os
print(os.listdir("../input"))
df_train = pd.read_csv("../input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
df_train.head()
X = df_train.iloc[:, 2].values
y = df_train.iloc[:, 1].values
df_test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", quoting=3)
X_1 = df_test.iloc[:, 1].values
def review_to_words(raw_review):
    letters_only = re.sub("[^a-zA-Z]"," ", raw_review)
    lower_words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    words = [word for word in lower_words if word.isalpha()] #removing special character and numbers
    meaningful_words = [ w for w in words if not w in stops]
    return(" ".join(meaningful_words))
filtered_x = []
total_reviews = X.size  #total number of reviews present or number of rows
for i in range(0,total_reviews):
    filtered_x.append(review_to_words(X[i]))
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split( filtered_x, y, test_size = 0.2, random_state = 0)
x_test = df_test["review"].map(review_to_words)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=2000) #tokenised to 2000 most frequent words
tokenizer.fit_on_texts(filtered_x)
# padding sequence to the limit is 500 words so it will look 500 words back 
train_reviews_tokenized = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(train_reviews_tokenized, maxlen=400)
val_review_tokenized = tokenizer.texts_to_sequences(X_val)
X_val = pad_sequences(val_review_tokenized, maxlen=400)
test_review_tokenized = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(test_review_tokenized, maxlen=400)
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
model = Sequential()
model.add(Embedding(20000,128)) #20000 words and funneling them into 128 hidden neurons
model.add(LSTM(128,dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(1, activation = "sigmoid"))
#compiling model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size = 32, epochs = 5, validation_data=[X_val, Y_val])
prediction = model.predict(x_test)
y_pred = (prediction > 0.5)
df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
y_test = df_test["sentiment"]
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)
