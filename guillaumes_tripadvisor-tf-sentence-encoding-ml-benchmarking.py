import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS 



import tensorflow as tf

import tensorflow_hub as hub

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

import time
run_wordcloud = True

max_data_size = None #for modeling

FILE_PATH = '/kaggle/input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv'

df = pd.read_csv(FILE_PATH)
df.head()
df.shape
df.isna().sum()
# functions come from 

# https://towardsdatascience.com/plotting-probabilities-for-discrete-and-continuous-random-variables-353c5bb62336

def frequencies(values):

    frequencies = {}

    for v in values:

        if v in frequencies:

            frequencies[v] += 1

        else:

            frequencies[v] = 1

    return frequencies



def probabilities(sample, freqs):

    probs = []

    for k,v in freqs.items():

        probs.append(round(v/len(sample),1))

    return probs



ratings = df.Rating.tolist()

freqs = frequencies(ratings)

probs = probabilities(ratings, freqs)

x_axis = list(set(ratings))

plt.bar(x_axis, freqs.values())
plt.bar(x_axis, probs)
# Python program to generate WordCloud 

# code comes from https://www.geeksforgeeks.org/generating-word-cloud-python/
def plot_wordcloud(wordcloud):

    # plot the WordCloud image                        

    plt.figure(figsize = (8, 8), facecolor = None) 

    plt.imshow(wordcloud) 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 



    plt.show()



def wordcloud_from_sentences(sentences):

    comment_words = '' 

    stopwords = set(STOPWORDS)



    # iterate through the csv file 

    for val in sentences: 



        # typecaste each val to string 

        val = str(val) 



        # split the value 

        tokens = val.split() 



        # Converts each token into lowercase 

        for i in range(len(tokens)): 

            tokens[i] = tokens[i].lower() 



        comment_words += " ".join(tokens)+" "



    wordcloud = WordCloud(

        width = 800, 

        height = 800, 

        background_color ='white',             

        stopwords = stopwords, 

        min_font_size = 10

    ).generate(comment_words)

    

    plot_wordcloud(wordcloud=wordcloud)
# wordcloud with all reviews

if run_wordcloud:

    wordcloud_from_sentences(sentences=df.Review)
if run_wordcloud:

    df_good_review = df[df['Rating'] >= 4]

    wordcloud_from_sentences(sentences=df_good_review.Review)
# wordcloud with all reviews

if run_wordcloud:

    df_bad_review = df[df['Rating'] < 4]

    wordcloud_from_sentences(sentences=df_bad_review.Review)
df.Rating = df.Rating.astype('int')

train, test = train_test_split(df.head(max_data_size), test_size=0.1)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
%%time

X_train = embed(train.Review.to_numpy())

X_test = embed(test.Review.to_numpy())

y_train = train.Rating.to_numpy()

y_test = test.Rating.to_numpy()
def fit_lgb(X_train, y_train, model):

        #d_train = lgb.Dataset(X_train, label=y_train)

        param = {'num_leaves': 31, 'objective': 'binary'}

        param['metric'] = 'auc'

        model = lgb.LGBMClassifier()

        model = model.fit(X_train, y_train)

        return model





def fit(X_train, y_train, model_str, model):

    if model_str == 'lgb':

        model = fit_lgb(X_train, y_train, model)

    elif model_str == 'Keras':

        model.fit(X_train, y_train, epochs=1)

    else:

        model.fit(X_train, y_train)

    return model





def evaluate(model_str, model, X_train, y_train, X_test, y_test):

    model = fit(X_train, y_train, model_str, model)

    y_pred = model.predict(X_test)

    print(f'Y_pred={y_pred}\nY_test={y_test}')

    acc = accuracy_score(y_pred, y_test)

    print(f'acc: {acc}')

    return acc





def keras_model(X_train):

    inputs = tf.keras.Input(shape=(X_train.shape[1],))

    x = tf.keras.layers.Dense(10, activation='relu')(inputs)

    outputs = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
result_dict = {}

dict_model = {'SGDClassifier': SGDClassifier(),

              'Keras': keras_model(X_train),

              'RandomForestClassifier': RandomForestClassifier(n_estimators=1),

              'lgb': lgb,

              'RidgeClassifier': RidgeClassifier()}



for model_str, model in dict_model.items():

    start = time.time()

    acc = evaluate(

        model_str, model, X_train, y_train, X_test, y_test

    )

    result_dict[model_str] = acc

    print(f'{model_str} runs in {round(time.time()-start, 2)}s.')
result_dict = {

    k: v

    for k, v in sorted(result_dict.items(),

                       key=lambda item: item[1],

                       reverse=True)

}

result_dict