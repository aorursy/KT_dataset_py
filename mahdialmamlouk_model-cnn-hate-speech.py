import numpy as np

import pandas as pd

import nltk

from wordcloud import WordCloud

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



data = pd.read_csv('../input/hate-speech-twitter-train-and-test/train_E6oV3lV.csv')

data
def eval_fun(labels, preds):

    labels = label.split(' ')

    preds = tweet.split(' ')

    rr = (np.intersect1d(label, tweet))

    precision = np.float(len(rr)) / len(tweet)

    recall = np.float(len(rr)) / len(label)

    try:

        f1 = 2 * precision * recall / (precision + recall)

    except ZeroDivisionError:

        return (precision, recall, 0.0)

    return (precision, recall, f1)

print(1)
import numpy as np

print("Hatred labeled: {}\nNon-hatred labeled: {}".format(

    (data.label == 1).sum(),

    (data.label == 0).sum()

))
hashtags = data['tweet'].str.extractall('#(?P<hashtag>[a-zA-Z0-9_]+)').reset_index().groupby('level_0').agg(lambda x: ' '.join(x.values))

data.loc[:, 'hashtags'] = hashtags['hashtag']

data['hashtags'].fillna('', inplace=True)



data.loc[:, 'mentions'] = data['tweet'].str.count('@[a-zA-Z0-9_]+')



data.tweet = data.tweet.str.replace('@[a-zA-Z0-9_]+', '')
data.tweet = data.tweet.str.replace('[^a-zA-Z]', ' ')
from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

from nltk import pos_tag, FreqDist, word_tokenize



stemmer = SnowballStemmer('english')

lemmer = WordNetLemmatizer()



part = {

    'N' : 'n',

    'V' : 'v',

    'J' : 'a',

    'S' : 's',

    'R' : 'r'

}



def convert_tag(penn_tag):

    if penn_tag in part.keys():

        return part[penn_tag]

    else:

        return 'n'





def tag_and_lem(element):

    sent = pos_tag(word_tokenize(element))

    return ' '.join([lemmer.lemmatize(sent[k][0], convert_tag(sent[k][1][0]))

                    for k in range(len(sent))])

    



data.loc[:, 'tweet'] = data['tweet'].apply(lambda x: tag_and_lem(x))

data.loc[:, 'hashtags'] = data['hashtags'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

texts = np.array(data['tweet'])

labels = np.array(data['label'])
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



maxlen = 100

samples = texts.shape[0]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index



data = pad_sequences(sequences, maxlen)

print(data.shape)

print(labels.shape)

x_train = data[:7925]

y_train = labels[:7925]



x_val = data[7925:8925]

y_val = labels[7925:8925]



x_test = data[8925:]

y_test = labels[8925:]



print(x_train.shape)

print(x_val.shape)

print(x_test.shape)

##





from keras.models import Sequential

from keras.layers import Dense, Embedding, SimpleRNN, LSTM, GRU, Bidirectional

model = Sequential()

model.add(Embedding(1+len(word_index), 16))

model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.5, return_sequences=True)))

model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.5)))

model.add(Dense(1, activation='sigmoid'))

model.summary()





model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train,y_train,epochs=10,batch_size=64,validation_data=(x_val,y_val))

def cnfmatrix(y_test,results):

    fp = 0.0

    fn = 0.0

    tp = 0.0

    tn = 0.0

    t = 0.0

    n = 0.0

    results.shape

    for i in range(results.shape[0]):

        if y_test[i]==1 and results[i]==1:

            tp+=1

            t+=1

        elif y_test[i]==1 and results[i]==0:

            fn+=1

            t+=1

        elif y_test[i]==0 and results[i]==1:

            fp+=1

            n+=1

        elif y_test[i]==0 and results[i]==0:

            tn+=1

            n+=1

    print(tp/results.shape[0],fp/results.shape[0])

    print(fn/results.shape[0],tn/results.shape[0])

    Precision  = tp/(tp+fp)

    Recall = tp/(tp+fn)

    print("Precision: ",Precision,"Recall: ",Recall)

    f1score = (2*Precision*Recall)/(Precision+Recall)

    print("f1score: ",f1score)

    print("accuracy: ",(tp+tn)/results.shape[0])

    print("hate_acc: ", (tp)/t)

    print("non_hate_acc: ", (tn)/n)
predictions = model.predict(x_test)
results = []

for prediction in predictions:

    if prediction < 0.5:

        results.append(0)

    else:

        results.append(1)

        

results = np.array(results)
cnfmatrix(y_test, results)

import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
def test(model):

    f1_score = lambda precision, recall: 2 * ((precision * recall) / (precision + recall))

    nexamples, recall, precision = model.test('fasttext.test')

    print (f'recall: {recall}' )

    print (f'precision: {precision}')

    print (f'f1 score: {f1_score(precision,recall)}')

    print (f'number of examples: {nexamples}')

print(1)
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])#

#history = model.fit(x_train,y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
##from wordcloud import WordCloud, STOPWORDS

##stopwords = STOPWORDS.add('amp')

##traindata = ['tweet', 'label']

##all_words = ' '.join('tweet', 'label')

##hatred_words = ' '.join(data[data.label == 1].tweet.values)



##plt.figure(figsize=(16, 8))



##cloud1 = WordCloud(width=400, height=400, background_color='white', stopwords=stopwords).generate(all_words)

##plt.subplot(121)

##plt.imshow(cloud1, interpolation="bilinear")

#plt.axis("off")

#plt.title('All tweets', size=20)



##cloud2 = WordCloud(width=400, height=400, background_color='white', stopwords=stopwords).generate(hatred_words)

##plt.subplot(122)

##plt.imshow(cloud2, interpolation="bilinear")

##plt.axis("off")

##plt.title('Hatred tweets', size=20)

##plt.show()
##all_hashtags = FreqDist(list(' '.join(data.hashtags.values).split())).most_common(10)

##hatred_hashtags = FreqDist(list(' '.join(data[data.label==1].hashtags.values).split())).most_common(10)

##plt.figure(figsize=(14, 6))

##ax = plt.subplot(121)

##pd.DataFrame(all_hashtags, columns=['hashtag', 'Count']).set_index('hashtag').plot.barh(ax=ax, fontsize=12)

##plt.xlabel('# occurrences')

##plt.title('Hashtags in all tweets', size=13)

##ax = plt.subplot(122)

##pd.DataFrame(hatred_hashtags, columns=['hashtag', 'Count']).set_index('hashtag').plot.barh(ax=ax, fontsize=12)

##plt.xlabel('# occurrences')

##plt.ylabel('')

##plt.title('Hashtags in hatred tweets', size=13)

##plt.show()
##print("Number of mentions: {}\nNumber of tweets having a mention: {}\nCorrelation with label: {}".format(

##    data.mentions.sum(),

##    len(data[data.mentions > 0]),

##    np.corrcoef(data.mentions, data.label)[0][1]

##))
##data.drop('mentions', axis=1, inplace=True)
##from sklearn.feature_extraction.text import TfidfVectorizer

##from nltk.corpus import stopwords



##vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 3), min_df=10)

##features = vectorizer.fit_transform(data.tweet)
##from sklearn.linear_model import LogisticRegression

##from sklearn.svm import SVC

##from sklearn.model_selection import train_test_split, GridSearchCV

##from sklearn.metrics import f1_score



##x_train, x_test, y_train, y_test = train_test_split(features, data.label)
##params = {'penalty': ['l1', 'l2'], 'C': [3, 10, 30, 100, 300]}

##lrmodel = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=150), param_grid=params, scoring='f1', cv=5, n_jobs=-1)

##lrmodel.fit(X_train, y_train)

##print("Best parameters found were {} with F1 score of {:.2f}".format(

##    lrmodel.best_params_,

##    lrmodel.best_score_

##))

##probas = lrmodel.predict_proba(X_test)

##thresholds = np.arange(0.1, 0.9, 0.1)

##scores = [f1_score(y_test, (probas[:, 1] >= x).astype(int)) for x in thresholds]

##plt.plot(thresholds, scores, 'o-')

##plt.title("F1 score for different thresholds")

##plt.ylabel("Score")

##plt.xlabel("Threshold")

##plt.show()
##params = {'C': [1000, 3000, 9000, 15000]}

##svc = GridSearchCV(SVC(kernel='rbf', gamma='auto'), param_grid=params, scoring='f1', cv=3, n_jobs=-1)

##svc.fit(X_train, y_train)

##print("Best parameters found were {} with F1 score of {:.2f}".format(

##    svc.best_params_,

##    svc.best_score_

##))

##predictions = svc.predict(X_test)

##print("\nF1 test score for SVC: {:.2f}".format(f1_score(y_test, predictions)))