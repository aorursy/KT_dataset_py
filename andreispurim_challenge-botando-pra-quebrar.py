import numpy

import pandas

%matplotlib inline



Reviews = pandas.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

print(Reviews.shape)

print(Reviews.head())
import spacy



Spacy = spacy.load('en_core_web_sm')



Chosen_Sentence = Spacy(Reviews['review'][1])

for i,word in zip(range(20),Chosen_Sentence):

    print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')
from spacy import displacy



displacy.render(Chosen_Sentence, style='dep', jupyter=True, options={'distance': 50})
for entity in Chosen_Sentence.ents:

    print(entity.text + ' - ' + entity.label_ + ' - ' + str(spacy.explain(entity.label_)))



displacy.render(Chosen_Sentence, style='ent', jupyter=True)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC, SVC, NuSVC

from sklearn.naive_bayes import MultinomialNB, GaussianNB

import matplotlib.pyplot as matplotlib

from matplotlib.lines import Line2D

from xgboost import XGBClassifier

import seaborn

import time

import re





# First, let's make a small function to clean our strings, because as we have seen before, there are tons of unwanted punctuations and other useless tags



def clear_sentence(sentence: str) -> str:

    '''A function to clear texts using regex.'''

    sentence = re.sub(r'\W', ' ', str(sentence))

    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)

    sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence) 

    sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

    sentence = re.sub(r'^b\s+', '', sentence)

    sentence = sentence.lower()

    return sentence





# Now, let's get a small sample of our reviews

Reviews_small = Reviews[0:5000]

Reviews_small



# Makes two datasets, x and y, x will be the clear reviews and y will be the sentiment

x = [clear_sentence(sentence) for sentence in Reviews_small['review']]

y = Reviews_small['sentiment'].tolist()



# Split the dataset in a 80%/20% fashion

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)





# I'm making two dictionaries, one for models to transform our words in vectors and the other of models to work on these vectors

Vectorizer_Models = {

    'Count': CountVectorizer(stop_words="english"),

    'Hash': HashingVectorizer(stop_words="english"),

    'Tfidf': TfidfVectorizer(stop_words="english",ngram_range=(1, 2))

}



ML_Models = {

    'LinearSVC': LinearSVC(),

    'SVC': SVC(),

    'NuSVC': NuSVC(),

    'DecisionTree': DecisionTreeClassifier(),

    'XGBClassifier': XGBClassifier(),

    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=0),

    'SGDC': SGDClassifier(),

    'MultiNB': MultinomialNB(),    

}



# I'll make a new list for a future dataset to see the accuracy and time of each algorithm.

plotting_data = []



for j,vector in enumerate(Vectorizer_Models):

    

    # Let's first vectorize, because the vectorized words will be used in common by all MLs. Also, starts counting the time to vectorize.

    time_vector_start = time.time()

    X_train_vectorized = Vectorizer_Models[vector].fit_transform(X_train) 

    X_test_vectorized= Vectorizer_Models[vector].transform(X_test)

    time_vector_end = time.time()

    

    for i,ml in enumerate(ML_Models):

        

        # Small detail: Multinomial Naive-Baise does not work with negative numbers, so we can just use him with Count

        if (ml == 'MultiNB' and vector != 'Count') == False:

            # Ok, let's start the time and put our models to fit the data.

            starting_time = time.time()

            model = ML_Models[ml]

            model.fit(X_train_vectorized, y_train)



            # Predict the data and try to find the accuracy

            y_predicted = model.predict(X_test_vectorized)

            accuracy = accuracy_score(y_test, y_predicted)

            ending_time = time.time()



            # Now, get the times and append everything in our plotting data.

            cut_time = round(time_vector_end - time_vector_start,2)

            ml_time = round(ending_time - starting_time,2)

            plotting_data.append([ml,vector,accuracy,ml_time,cut_time,cut_time+ml_time])





# Makes a pandas dataset for our data (for better visualization)

plot_times = pandas.DataFrame(plotting_data, columns=['ML','Vectorizer','Accuracy','ML_Time','Cut_time','Total_time'])



# Now, let's make a Seaborn scatterplot

seaborn.set(color_codes=True)

matplotlib.figure(figsize=(9, 6))

matplotlib.title("Best vectorization and Accuracy Algorithms")



ax = seaborn.scatterplot(data=plot_times, x='Total_time', y='Accuracy', hue='ML', style='Vectorizer')

matplotlib.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

ax.set(xlabel="Time (s)", ylabel="Accuracy")

plot_times
x = [clear_sentence(sentence) for sentence in Reviews['review']]

y = Reviews['sentiment'].tolist()



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

X_train_vectorized = Vectorizer_Models[vector].fit_transform(X_train) 

X_test_vectorized= Vectorizer_Models[vector].transform(X_test)



Chosen_Models = {

    'LinearSVC': LinearSVC(),

    'SGDC': SGDClassifier(),

}



for ml in Chosen_Models:

    starting_time = time.time()

    model = Chosen_Models[ml]

    model.fit(X_train_vectorized, y_train)

    y_predicted = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_predicted)

    ending_time = time.time()

    print(ml,'Accuracy:',"{:.2f}".format(accuracy*100),"in","{:.2f}s".format(ending_time-starting_time))

    print(confusion_matrix(y_test, y_predicted))
# Insert fine tuning
from keras.layers.core import Activation, Dropout, Dense

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.layers.embeddings import Embedding

from keras.preprocessing.text import one_hot

from keras.layers import GlobalMaxPooling1D

from keras.models import Sequential

from keras.layers import Flatten



tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(X_train)



X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)



vocab_size = len(tokenizer.word_index) + 1



maxlen = 100



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



# model = Sequential()

# model.add(tf.keras.layers.Dense(128,input_shape=(X_train[0].shape)))

# model.add(tf.keras.layers.Dense(512, activation='relu'))

# model.add(tf.keras.layers.BatchNormalization())

# model.add(tf.keras.layers.Dense(128, activation='relu'))

# model.add(tf.keras.layers.Dense(512, activation='relu'))

# model.add(Dropout(0.25))

# model.add(tf.keras.layers.Dense(46, activation='softmax'))

# print(model.summary())
# Load the data and take a look at how tweet datasets usually look like

Airlines_Total = pandas.read_csv('../input/twitter-airline-sentiment/Tweets.csv')



# Now, we won't be using any other data other than the text and the sentiment. We could use location or reason for a in-depth analysis but we'll reserve that to the Australian Tweets.

Airlines = Airlines_Total[['airline_sentiment','text']]



# Use our clear_sentence function made at the very beggining

x = [clear_sentence(sentence) for sentence in Airlines['text']]

y = Airlines['airline_sentiment'].tolist()



# Let's use the SVC model we used before.

vector = TfidfVectorizer(stop_words="english",ngram_range=(1, 2))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

X_training = vector.fit_transform(X_train) 

X_testing = vector.transform(X_test)

model = LinearSVC()

model.fit(X_training, y_train)

y_prediction = model.predict(X_testing)

accuracy = accuracy_score(y_test, y_prediction)

print('Accuracy:',"{:.2f}".format(accuracy*100))



# I'm adding the head after because the print obscures it.

Airlines_Total.head()
# Insert deeplearning here
Australia_Tweets = pandas.read_csv('../input/australian-election-2019-tweets/auspol2019.csv')

Australia_Tweets.head()
Australia_geocode = pandas.read_csv('../input/australian-election-2019-tweets/location_geocode.csv')

Australia_geocode.columns = ['user_location','lat','long']

Tweets_only = Australia_Tweets[['full_text','user_name','user_location']]



Australia = pandas.merge(Tweets_only, Australia_geocode)

Australia
Chosen_Sentence = Spacy(Tweets_in_Australia['full_text'][5])

for i,word in zip(range(20),Chosen_Sentence):

    print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')
def get_region(data, bot_lat, top_lat, left_lon, right_lon):

    top = data.lat <= top_lat

    bot = data.lat >= bot_lat

    left = data.long >= left_lon

    right = data.long <= right_lon

    index = top&bot&left&right 

    return data[index]

    

Tweets_in_Australia = get_region(Australia,-44,-10,109,156)
election_text = [clear_sentence(sentence) for sentence in Tweets_in_Australia['full_text']]



election_predicting = vector.transform(election_text)

feelings = model.predict(election_predicting)

feelings
feelingsdf = pandas.DataFrame(feelings,columns=['feeling'])

Australia_predicted = pandas.concat([Tweets_in_Australia, feelingsdf], axis=1).dropna()

Australia_predicted
from mpl_toolkits.basemap import Basemap



Australia_map = Basemap(urcrnrlat=-10,llcrnrlat=-44,llcrnrlon=109,urcrnrlon=156)

matplotlib.figure(figsize=(12,10))

Australia_map.bluemarble(alpha=0.9)

seaborn.scatterplot(x='long', y='lat', hue='feeling', data=Australia_predicted)



matplotlib.title("Tweets about the Australian Election by location")

matplotlib.show()