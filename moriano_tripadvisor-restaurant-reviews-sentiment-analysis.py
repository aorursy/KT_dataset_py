import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
raw = pd.read_csv("../input/londonbased-restaurants-reviews-on-tripadvisor/tripadvisor_co_uk-travel_restaurant_reviews_sample.csv")
raw.head()
cleaned = raw[["review_text", "rating"]]
cleaned.head()
set(cleaned["rating"])
def clean(raw_data):
    as_string = str(raw_data["rating"])
    try:
        return int(as_string[0]) # Our number of bubbles is simply the first character
    except:
        # Some values cannot be converted... in which case set to -1 and we will remove them later
        return -1

cleaned["y"] = cleaned.apply(clean, axis=1)
cleaned.head()
print(cleaned["review_text"][0], "\n")
print(cleaned["review_text"][1], "\n")
print(cleaned["review_text"][2])
import re
def clean_text(df):
    text = str(df["review_text"])
    text = text.lower()
    text = re.sub("[^a-z\s]", "", text)
    return text

cleaned["review_text"] = cleaned.apply(clean_text, axis=1)
print(cleaned["review_text"][0], "\n")
print(cleaned["review_text"][1], "\n")
print(cleaned["review_text"][2], "\n")
del cleaned["rating"]
cleaned.columns = ["X", "y"]
cleaned.head()
cleaned = cleaned[cleaned["y"] > 0]
cleaned.head()
cleaned["review_size"] = cleaned["X"].str.count(" ")
cleaned["review_size"].describe()
cleaned["review_size"].plot(title="Number of words per review", figsize=(20, 8))
cleaned = cleaned[cleaned["review_size"] >= 50]
cleaned["review_size"].describe()
del cleaned["review_size"]
cleaned.describe()
scores = cleaned.groupby(["y"]).agg("count")
scores.plot(kind='bar', title="Review by number of stars", figsize=(10, 7))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Lets tokenize the works, we will only keep the 20.000 most common words
VOCAB_SIZE=20000
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(cleaned["X"])
sequences = tokenizer.texts_to_sequences(cleaned["X"])
print("Number of sequences", len(sequences))
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print("Index for 'great' is ", word_index["great"])
SEQUENCE_SIZE=120
data = pad_sequences(sequences, maxlen=SEQUENCE_SIZE)
print("Our padded data has a shape of ", data.shape)
labels = cleaned["y"]
# Normalize the labels to values between 0 and 1, this can be done by simply dividing by 5, we will later need to multiply our predictions by 5
labels = labels / 5

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
validation_split = int(data.shape[0] * 0.7)
X_train = data[0: validation_split]
y_train = labels[0: validation_split]

X_test = data[validation_split:]
y_test = labels[validation_split:]
print("X_train", X_train.shape, "y_train", y_train.shape)
print("X_test", X_test.shape, "y_test", y_test.shape)
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed, Embedding, Dropout
EMBEDDING_DIM = 100
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SEQUENCE_SIZE))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(1, activation="linear"))
from keras.optimizers import RMSprop

rmsprop = RMSprop(lr=0.0001)

model.compile(loss='mae', optimizer=rmsprop)
for layer in model.layers:
    print("Layer", layer.name, "Is trainable? ==>", layer.trainable)
model.summary()
history = model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.3)
model_history = pd.DataFrame.from_dict(history.history)
model_history
model_history[["loss", "val_loss"]].plot(title="Learning curve, loss", figsize=(17, 6))
model.evaluate(X_test, y_test)
five_star_review = """My wife and I were here for dinner last night, what an amazing experience. Entree were crab and pork, both amazing dishes full of flavor. I understand what others have said in their reviews re: portion size, however; the skill involved in making these dishes deserves somebody to savour it and not rush it. Mains were the wagyu, and the lamb. Despite both being meat dishes, I couldn't say which one I liked more despite trying both as they were so different. Full of flavours, expertly cooked. The wagyu was tender and the flavours amazing, the lamb was perfect and all the accompaniments were expertly suited. Dessert was a chocolate mousse, you only need a little bit because it's so rich. The whole experience was amazing. A special mention to Ashe who had the most amazing/funny/professional table manner when serving, he's a true asset to the establishment. Unfortunately cash flow has been tight this month for good reasons, but it meant we weren't able to tip after receiving the bill. Despite this, we hope to return again and will make up for our evening last night on our next visit."""
three_star_review = """We had heard good things about this place, but it has let us down twice. The first time it was busy and we gave up waiting for service after about 20 minutes. So, yesterday we tried again with 6 friends who were visiting the 'Mountains.' Sadly, the waiter seemed unaware of almost anything on the menu, got four orders wrong and failed to even place one order at all! He then went home and another server had to deal with the mistakes. Four of us were disappointed with the food and the other four thought it was just okay. The location is great but this cafe needs a real shake-up."""
one_star_review = """The location was spectacular, the fireplace created a beautiful ambiance on a winters day. However the food was a lot to be desired. My friend had been previously with her husband and enjoyed the meals, however what we got was disgusting! We ordered sausages from the local butcher with polenta....well Coles cheap beef sausages are a lot tastier and moist, they looked like they boiled them and put them on a plate, absolutely no grill marks, the polenta was so dry I gagged. My other friend ordered their homemade pie that was stuck in a microwave oven to be heated and ended with soggy pastry. Sorry but definitely need someone new in the kitchen. The staff, well the first lady that took us to our table was lovely, but the second person we ordered our coffees with was so upset that we were going to delay her from leaving work on time. Definitely won't be returning or recomending!"""

texts = [five_star_review, three_star_review, one_star_review]

text_data = pd.DataFrame(data=texts, columns=["review_text"])
text_data["review_text"] = text_data.apply(clean_text, axis=1)
sequences_test = tokenizer.texts_to_sequences(text_data["review_text"])
to_predict = pad_sequences(sequences_test, maxlen=SEQUENCE_SIZE)

output = model.predict(to_predict, verbose=False)
print("Predicted", output*5)
print(os.listdir("../input"))
print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join("../input/glove-global-vectors-for-word-representation", 'glove.6B.100d.txt'))
count = 0
for line in f:
    count += 1
    if count % 100000 == 0:
        print(count, "vectors so far...")
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
print("Word 'great' has a vector like this \n", embeddings_index["great"])
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print("Size of the embedding matrix is ", embedding_matrix.shape)
embedding_layer = Embedding(len(word_index) + 1,  # Input dim
                            EMBEDDING_DIM,  # Output dim
                            weights=[embedding_matrix],
                            input_length=SEQUENCE_SIZE,  
                            trainable=False)

# This layer will take an input of 
#           BatchSize x SEQUENCE_SIZE
# And the output will be
#           None, SEQUENCE_LENGTH, 100

from keras.layers import Flatten, TimeDistributed
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.3))
model.add(LSTM(128))
model.add(Dropout(0.3))
model.add((Dense(1, activation='linear')))
rmsprop = RMSprop(lr=0.001)

model.compile(loss='mae', optimizer=rmsprop)

for layer in model.layers:
    print("Layer", layer.name, "Is trainable? ==>", layer.trainable)
model.summary()

history = model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.3)
model_history = pd.DataFrame.from_dict(history.history)
model_history
model_history[["loss", "val_loss"]][model_history["loss"]< 3].plot(title="Learning curve, loss", figsize=(17, 6))
output = model.predict(to_predict, verbose=False)
print("Predicted", output*5)
model.evaluate(X_test, y_test)
