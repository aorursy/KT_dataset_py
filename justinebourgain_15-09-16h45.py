from keras.preprocessing import sequence
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
y_train = newsgroups_train.target

X_test = newsgroups_test.data
y_test = newsgroups_test.target
new_X=X_test.copy()
X_train[0]
top_words = 5000
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train[0]
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=max_words, padding='post')
from keras.utils import to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)
model = Sequential()

model.add(Embedding(5000, 26, input_length=500))
model.add(Conv1D(26,26, activation='relu'))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20, batch_size=128, verbose=2)

from keras.models import Model
encoder = Model(model.input, model.layers[-2].output)
encoder.summary()
encoded_imgs = encoder.predict(X_test)
X_test.shape
X_test[2]
newsgroups_test.data[2]
img_to_find = encoded_imgs[2]
def custom_cosine_sim(a,b):
    return np.dot(a, b) / ( np.linalg.norm(a) * np.linalg.norm(b))
from scipy import spatial
cosine_list = []
for index_image,xt in enumerate(encoded_imgs):
    #print (spatial.distance.cosine(img_to_find, xt))
    #print (1 - spatial.distance.cosine(img_to_find, xt))
    #print (custom_cosine_sim(img_to_find, xt))
    #print()
    result = 1 - spatial.distance.cosine(img_to_find.reshape(-1), xt.reshape(-1))
    cosine_list.append(dict({'res':result, 'i':index_image}))
from operator import itemgetter
cosine_list.sort(key=itemgetter('res'), reverse=True)
cosine_list


for i in range(10):
    print(newsgroups_test.data[cosine_list[i]['i']])
   


top_words = 5000
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(new_X)
new_X = tokenizer.texts_to_sequences(new_X)
