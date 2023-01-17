import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import re, string

import matplotlib.pyplot as plt
dataset = '../input/news-category-dataset/News_Category_Dataset_v2.json'

GloVe_text_file = '../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt'

DIM = 50
data = pd.read_json(dataset, lines=True)[['category','headline','authors','short_description']]

data.head()
embeddings = {}



with open(GloVe_text_file,'r') as file:

    line = file.readline().split()                    # Split line by whitespace

    while not line == []:                             # Until EOF

        word = line[0]                                  # Parse the word

        vec = np.array(line[1:]).astype(np.float32)     # Parse the vector

        embeddings[word] = vec                               # Add word-vec combo to embeddings dictionary

        line = file.readline().split()                  # goto next line

        

def embed(word):

    if word in embeddings.keys():

        return embeddings[word]

    else:

        return np.zeros(DIM)
# Find top 50 authors

_ = data[['authors','category']].groupby('authors').count().sort_values('category',ascending =False).index[:49] #[:50]



# Create one_hot encodings for top 50

top_50_authors = {}

i_50 = np.identity(50).astype(np.float32)

for i,author in enumerate(_):

    top_50_authors[author] = i_50[i,:]



def get_author_encoding(row):

    author = row['authors']

    if author in top_50_authors.keys():

        return top_50_authors[author]

    else:

        return i_50[-1,:] # np.zeros(50)
CATEGORIES = {}

categories = data['category'].unique()

for i,c in enumerate(categories):

    zeros = np.zeros(len(categories))

    zeros[i] = 1

    CATEGORIES[c] = zeros



cat_vec2txt = {}

for i in list(CATEGORIES.keys()):

    cat_vec2txt[CATEGORIES[i].argmax()] = i
# TRAIN-TEST SPLIT

labels = data['category'].values

categories = set(labels)

train_idxs = np.array([])

test_idxs = np.array([])

train_distribution = []

for c in categories:

    subset = np.where(labels==c)[0]

    np.random.shuffle(subset)

    q = subset.shape[0]*8//10

    train_idxs = np.hstack((train_idxs,subset[:q]))

    test_idxs = np.hstack((test_idxs,subset[q:]))

    train_distribution.append(q)





train_data = data.iloc[train_idxs]

test_data = data.iloc[test_idxs]   

    

    

print('Train Shape: {}'.format(train_idxs.shape))

print('Test Shape: {}'.format(test_idxs.shape))
categories, train_distribution = zip(*sorted(list(zip(categories, train_distribution)),key=lambda x: x[1], reverse = True))

counts = data[['authors','category']].groupby('category').count().sort_values('authors',ascending =False).values[:,0]

plt.bar(categories, counts, label= 'Total')

plt.bar(categories, train_distribution, label= 'Training Set')

plt.title('train_distribution')

plt.xticks(rotation=270, size=7)

plt.legend()

plt.show()
MAX_WORDS = 32



regex = re.compile('[^a-zA-Z ]')

def get_text_encoding(row):

    text = regex.sub('',row['headline'] + ' ' + row['short_description']).lower().split()

    

    word_matrix = np.zeros((MAX_WORDS,DIM+1))

    for i in range(MAX_WORDS):

        if i<len(text):

            word_matrix[i] = np.append(embed(text[i]),0)

        else:

            v = np.zeros(DIM+1)

            v[-1] = 1

            word_matrix[i] = v

    return word_matrix



def get_input_matrix(df, idx):

    row = df.iloc[idx]

    

    word_matrix = get_text_encoding(row)

    author = get_author_encoding(row)

    author_matrix = np.zeros((word_matrix.shape[0],author.shape[0])) + author

    matrix = np.hstack((author_matrix,word_matrix))

    

    cat_vec = CATEGORIES[row['category']]

    return matrix,cat_vec

    

    
for i in range(1):

    arr, cat = get_input_matrix(train_data, i)

    print(arr.shape)

    plt.imshow(arr)

    plt.title(cat_vec2txt[cat.argmax()])

    plt.xticks([])

    plt.show()
def generate_data(df, batch_size, shuffle = True):

    if shuffle:

        df = df.sample(frac=1).reset_index(drop=True)

    i = 0

    while True:

        image_batch = []

        category_batch = []

        for b in range(batch_size):

            if i == len(df):

                if shuffle:

                    df = df.sample(frac=1).reset_index(drop=True)

                i = 0

            image, category = get_input_matrix(df, i)

            image_batch.append(image)

            category_batch.append(category)

            i += 1



        yield np.array(image_batch), np.array(category_batch)
from keras.models import Sequential

from keras.layers import LSTM, Dense
BATCH_SIZE = 32

EPOCHS = 50



model = Sequential()

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(32, DIM+50+1)))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(41, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.summary()
history = model.fit_generator(

    generate_data(train_data, BATCH_SIZE),

    steps_per_epoch=6125,

    epochs=EPOCHS,

    validation_data = generate_data(test_data, BATCH_SIZE, shuffle=False),

    validation_steps = 1255

    )
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
def predict_gen(df, BATCH_SIZE):

    gen = generate_data(df, BATCH_SIZE, shuffle=False)

    global y_actual

    y_actual = []

    while True:

        x,y = next(gen)

        y_actual = y_actual + list(y)

        yield(x)

        

y_pred = model.predict(predict_gen(test_data, BATCH_SIZE), steps=1250)
# Convert One-Hot to integer

y_actual = np.array(y_actual)

y_pred = y_pred.argmax(1)

y_actual = y_actual.argmax(1)[:len(y_pred)] # Trim off the last batch from y_actual
from sklearn.metrics import confusion_matrix, accuracy_score





print('Accuracy: ', accuracy_score(y_actual, y_pred))

m = confusion_matrix(y_actual, y_pred, labels = list(np.arange(41)))



plt.figure(figsize=(20,10))

plt.imshow(m)

plt.yticks(list(np.arange(41)),list(CATEGORIES.keys()), size=10)

plt.xticks(list(np.arange(41)),list(CATEGORIES.keys()), size=10, rotation=270)

plt.ylabel("Actual")

plt.xlabel("Predicted")

plt.show()