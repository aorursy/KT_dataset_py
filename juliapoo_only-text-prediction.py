import os

print(os.listdir("../input"))

from tqdm import tqdm

tqdm.pandas()

import math



import numpy as np

import pandas as pd
#### Get data

train = pd.read_csv('../input/ndsc-beginner/train.csv')



# 1: beauty, 2: fashion, 3: mobile

def split_big_cats(train):

    bool1 = train['image_path'].str[0] == 'b'

    bool2 = train['image_path'].str[0] == 'f'

    bool3 = train['image_path'].str[0] == 'm'

    X1 = train[bool1]['title']

    X2 = train[bool2]['title']

    X3 = train[bool3]['title']

    y1 = train[bool1]['Category']

    y2 = train[bool2]['Category']

    y3 = train[bool3]['Category']

    return X1, X2, X3, y1, y2, y3



#### Get data only from the fashion set

X1, X2, X3, y1, y2, y3 = split_big_cats(train)

### Build word2vec



import re

import string

table = str.maketrans('','', string.punctuation)



def tokenize_text(text):

    text = text.translate(table)

    text = re.sub(r'\d+', '', text)

    return text.lower().split()



def process_title(title):

    word_list = tokenize_text(title)

    return word_list

    

def process_title_list(title_list):

    word_title_list = []

    for title in title_list:

        word_title_list.append(process_title(title))

    return word_title_list



from gensim.models import Word2Vec



def create_word2vec(X):

    wordlists = process_title_list(X)

    word2vec = Word2Vec(wordlists, min_count=4, size=128, seed = 1, workers = 10) 

    word2vec.train(wordlists, total_examples=len(wordlists), epochs=10)

    return word2vec



### Building the models

word2vec1 = create_word2vec(X1)

word2vec2 = create_word2vec(X2)

word2vec3 = create_word2vec(X3)
from gensim.models.doc2vec import Doc2Vec

doc2vec1 = Doc2Vec.load('../input/pretrained-doc2vec/doc2vec1.model')

doc2vec2 = Doc2Vec.load('../input/pretrained-doc2vec/doc2vec2.model')

doc2vec3 = Doc2Vec.load('../input/pretrained-doc2vec/doc2vec3.model')
#### Split data into test and train

from sklearn.model_selection import train_test_split

X1T, X1t, y1T, y1t = train_test_split(X1, y1, test_size = 10000, stratify = y1)

X2T, X2t, y2T, y2t = train_test_split(X2, y2, test_size = 10000, stratify = y2)

X3T, X3t, y3T, y3t = train_test_split(X3, y3, test_size = 10000, stratify = y3)
### Process the titles for doc2vec

def pandas_2_numpy(X):

        eh = []

        for i in X:

            eh.append(i)

        return np.array(eh)



#### Converts titles to vectors by doc2vec, then PCA (56 components), then normalise

def get_std(XT, doc2vec):

    # Titles to vector via doc2vec

    XT_title = XT.apply(lambda title : predict(title, doc2vec))

    XT_title = pd.DataFrame.from_items(zip(XT_title.index, XT_title.values)).transpose().fillna(value = 0)

    # Normalise everything such that the largest component has std=1, mean=0

    std = XT_title.std(axis = 0).max()

    # Returns std and PCA2 for the processing the test set

    return std



def process_titles1(XT, doc2vec, std):

    XT_title = XT.apply(lambda title : predict(title, doc2vec))/std

    XT_title = pd.DataFrame.from_items(zip(XT_title.index, XT_title.values)).transpose().fillna(value = 0)

    return XT_title



def predict(title, doc2vec):

    return doc2vec.infer_vector(tokenize_text(title))



std1 = get_std(X1, doc2vec1)

std2 = get_std(X2, doc2vec2)

std3 = get_std(X3, doc2vec3)
### Process the titles for word2vec

# Convert values to embeddings

def get_word_vec(word, word2vec):

    try:

        vec = word2vec.wv[word]

    except:

        vec = np.zeros(128)

    return vec



def title_to_array(title, word2vec):

    null = np.zeros(128)

    title = title.lower().split()[:50]

    title_vec_arr = [get_word_vec(x, word2vec) for x in title]

    title_vec_arr += [null] * (50 - len(title_vec_arr))

    return np.array(title_vec_arr)



def process_titles2(XT, word2vec):

    XT_title = np.array([title_to_array(title, word2vec) for title in XT])

    return XT_title
from sklearn import preprocessing

def get_lb(yT):

    lb = preprocessing.LabelBinarizer()

    lb.fit(yT)

    return lb



lb1 = get_lb(y1T)

lb2 = get_lb(y2T)

lb3 = get_lb(y3T)
def batch_gen(XT, yT, batch_size, word2vec, doc2vec, std, lb):

    num = math.floor(len(XT)/ batch_size)

    while True: 

        for i in range(num):

            titles = XT.iloc[i * batch_size: (i+1) * batch_size]

            XT_doc2vec = process_titles1(titles, doc2vec, std)

            XT_word2vec = process_titles2(titles, word2vec)

            yT_ = yT.iloc[i * batch_size: (i+1) * batch_size]

            yT_ = lb.transform(yT_)

            yield [XT_doc2vec, XT_word2vec], yT_

from keras.models import Model, load_model

from keras.layers import CuDNNLSTM, Dense, Bidirectional, Activation, Dropout, concatenate

from keras.engine.input_layer import Input

from keras.callbacks import EarlyStopping, ModelCheckpoint



def build_model(n):

    word2vec_input = Input(shape=(50, 128))

    x0 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(word2vec_input)

    x0 = Bidirectional(CuDNNLSTM(128))(x0)

    x0 = Dropout(0.05)(x0)

    doc2vec_input = Input(shape=(300,))

    x2 = concatenate(inputs = [x0, doc2vec_input], axis=-1)

    output = Dense(n, activation = 'softmax')(x2)



    model = Model(inputs = [doc2vec_input, word2vec_input], outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def visualise_model(model):

    from keras.utils import plot_model

    import matplotlib.image as mpimg

    import matplotlib.pyplot as plt

    plot_model(model, to_file='model.png')

    img1 = mpimg.imread('model.png')

    plt.figure(figsize = (10,10))

    plt.title(str(model))

    plt.imshow(img1)

    plt.show()
def build_and_train_model(XT, yT, Xt, yt, word2vec, doc2vec, std, lb, num_categories, epochs):

    model = build_model(num_categories)

    visualise_model(model)

    model.summary()

    

    gen = batch_gen(XT, yT, 128, word2vec, doc2vec, std, lb)

    Xt_doc2vec_input = process_titles1(Xt, doc2vec, std)

    Xt_word2vec_input = process_titles2(Xt, word2vec)

    

    earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')

    mcp_save = ModelCheckpoint('model_best'+str(num_categories)+'.h5', save_best_only=True, monitor='val_acc', mode='auto')

    hist = model.fit_generator(gen, epochs=epochs, steps_per_epoch=1000,

                               validation_data=([Xt_doc2vec_input, Xt_word2vec_input], 

                                                lb.transform(yt))

                               ,verbose=True, callbacks=[earlyStopping, mcp_save])

    return hist, load_model('model_best'+str(num_categories)+'.h5')
import warnings

warnings.filterwarnings("ignore")



hist1, model1 = build_and_train_model(X1T, y1T, X1t, y1t, word2vec1, doc2vec1, std1, lb1, 17, 30)

hist2, model2 = build_and_train_model(X2T, y2T, X2t, y2t, word2vec2, doc2vec2, std2, lb2, 14, 30)

hist3, model3 = build_and_train_model(X3T, y3T, X3t, y3t, word2vec3, doc2vec3, std3, lb3, 27, 30)
test = pd.read_csv('../input/ndsc-beginner/test.csv')



def y_pred2pred(y_pred, n):

    temp = [0] * n

    temp[y_pred.argmax()] = 1

    return np.array([temp])



def pred_from_title(sample):

    title = sample['title']

    bigcat = sample['image_path'][0]

    if bigcat == 'b':

        Xt_doc2vec = predict(title, doc2vec1)/std1

        Xt_word2vec = title_to_array(title, word2vec1)

        yt_ = model1.predict([np.array([Xt_doc2vec]), np.array([Xt_word2vec])])

        yt_ = y_pred2pred(yt_, 17)

        cat = lb1.inverse_transform(yt_)

    if bigcat == 'f':

        Xt_doc2vec = predict(title, doc2vec2)/std2

        Xt_word2vec = title_to_array(title, word2vec2)

        yt_ = model2.predict([np.array([Xt_doc2vec]), np.array([Xt_word2vec])])

        yt_ = y_pred2pred(yt_, 14)

        cat = lb2.inverse_transform(yt_)

    if bigcat == 'm':

        Xt_doc2vec = predict(title, doc2vec3)/std3

        Xt_word2vec = title_to_array(title, word2vec3)

        yt_ = model3.predict([np.array([Xt_doc2vec]), np.array([Xt_word2vec])])

        yt_ = y_pred2pred(yt_, 27)

        cat = lb3.inverse_transform(yt_)

    return cat



pred = []

for i in (range(len(test))):

    pred.append(pred_from_title(test.iloc[i])[0])



submit_df = pd.DataFrame({"itemid": test["itemid"], "Category": np.array(pred)})

submit_df.to_csv("submission.csv", index=False)

submit_df.head(5)