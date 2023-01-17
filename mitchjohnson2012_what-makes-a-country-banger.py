# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# import the data
path = r'/kaggle/input/data-on-songs-from-billboard-19992019/BillboardFromLast20/billboardHot100_1999-2019.csv'
bb_data = pd.read_csv(path, index_col=0)
bb_data.tail()
# Filter to only country songs
columns = ['Artists', 'Name', 'Weekly.rank', 'Peak.position', 'Lyrics']
country_songs = bb_data[bb_data.Genre.str.contains('Country', regex=False)][columns].copy()
country_songs
###
# Calculating Peak.position
###
from time import perf_counter
new_peak_positions = country_songs.groupby(['Artists','Name']).agg(min_rank=('Weekly.rank', min))


def newPeak(row):
    ndx = (row['Artists'], row['Name'])
    new_peak_position = new_peak_positions.loc[ndx, 'min_rank']
    if np.isnan(row["Peak.position"]):
        row["Peak.position"] = new_peak_position
    else:
        new_min = min(row["Peak.position"], new_peak_position)
        row["Peak.position"] = new_min
    return row["Peak.position"]

start = perf_counter()
country_songs["Peak.position"] = country_songs.apply(newPeak, axis=1)
stop = perf_counter()
print("Time to calculate new peak position:", stop-start) 
country_songs.head()
# Define the billboard rank for critical banger zone
highway_to_the_banger_zone = 5

# Separate by weekly rank
country_bangers = country_songs[bb_data['Weekly.rank'] <= highway_to_the_banger_zone]
country_bops = country_songs[bb_data['Weekly.rank'] > highway_to_the_banger_zone]

# import plotting modules
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# who has spent the most time on the charts?
bopzone_weeks = country_bops['Artists'].value_counts()[0:20]
fig, (ax1, ax2) = plt.subplots(2, figsize=(10,12))
ax1.set_title("Weeks in the Bop Zone (Top 20 Artists)")
sns.barplot(y=bopzone_weeks.index, x=bopzone_weeks.values, ax=ax1)

# who has spent the most time in banger zone?
bangerzone_weeks = country_bangers['Artists'].value_counts()[0:20]
ax2.set_title("Weeks in the Banger Zone (Top 20 Artists)")
sns.barplot(y=bangerzone_weeks.index, x=bangerzone_weeks.values, ax=ax2)
###
# Data Prep
### 

# Create a unique list of songs
columns = ['Artists', 'Name', 'Peak.position', 'Lyrics']
country_song_set = country_songs[columns].copy().drop_duplicates()

# Make sure Country Roads never made the charts
country_roads = country_song_set[country_song_set["Name"].str.contains("country road", case=False, regex=False)].loc[:,["Artists", "Name"]]
if country_roads.size == 0:
    print('"Take Me Home, Country Roads" did not make the billboards from 1999-2019')
else:
    print('"Take Me Home, Country Roads" did make the billboards from 1999-2019')
##
# Natural Language Processing - Introduction of terms
##

# Import NLP module
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# set up an example
test_string = country_song_set.loc[1, 'Lyrics']


# load the English language pipeline and tokenize the text string
nlp = spacy.load('en')
docx = nlp(test_string)

# demo of lemmatizing and stop words
lst = list()
dir(lst)
for word in docx:
    lst.append([word.text, word.lemma_, word.is_stop, word.is_punct, word.is_space])

# display the results
from tabulate import tabulate    
print(tabulate(lst[0:25], headers=['word', 'lemma', 'isStop', 'isPunct', 'isSpace']))

###
# Is Country Roads a banger? - Create the tokenizer
###

# Create a method to preprocess the lyrics
def Tokenize(lyrics):
    # Create doc
    doc = nlp(lyrics)
    
    #lemmatize and remove stop words
    tokens = [word.lemma_.lower().strip() for word in doc if word.lemma_ != '-PRON-' and not word.is_stop and not word.is_punct and not word.is_space]
    return tokens
    
print(Tokenize(test_string))
# Create real data set
X = country_song_set['Lyrics']
y = country_song_set['Peak.position'] <= highway_to_the_banger_zone

# Split the data into testing and validation sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# show sample from the train data
print('Train Data: IsBanger Counts\n---------------------------\n',y_train.value_counts())
print('\nTest Data: IsBanger Counts\n---------------------------\n',y_test.value_counts())
pd.concat([X_train, y_train], axis=1)
start = perf_counter()


# Use Bag of Words methodology (try TF-IDF later)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
bow_vector = CountVectorizer(tokenizer = Tokenize, ngram_range=(1,1))


# For binary target, use logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)


# Create the bow pipeline and fit the model
from sklearn.pipeline import Pipeline
bow_pipe = Pipeline([("Vectorizer", bow_vector),
                 ("Classifier", model)])
bow_pipe.fit(X_train, y_train)


# Predict values using the trained model
y_predicted_bow = bow_pipe.predict(X_test)


# Measure performance
stop = perf_counter()
print("Time to train and evaluate model (in seconds):", stop-start) 
# Create a confusion matrix
def ConfusionMatrix(lstPredictions, lstActual):
    tp = tn = fp = fn = 0
    for i in range(0, len(lstPredictions)):
        if lstPredictions[i] == True:
            if lstActual[i] == True: tp+=1
            else: fp+=1
        else:
            if lstActual[i] == False: tn+=1
            else: fn+=1
    return tp, tn, fp, fn


def EvalMetrics(tp, tn, fp, fn, display_results=True):
    # Model Accuracy - How often was the model right?
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    # Sensitivity/Recall/TP Rate - How often do we correctly classify the bangers?
    sens = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    # Specificity - How often did we correctly classify the bops?
    spec = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    # Precision - How certain are we that something that the model says is a banger is truly a banger?
    prec = tp / (tp + fp) if (tp + fp) != 0 else 0
    
    if display_results:
        print("Model Accuracy - How often was the model right? - {0:.0%}".format(acc))
        print("Sensitivity/Recall/TP Rate - How often do we correctly classify the bangers? - {0:.0%}".format(sens))
        print("Specificity - How often did we correctly classify the bops? - {0:.0%}".format(spec))
        print("Precision - How certain are we that something that the model says is a banger is truly a banger? - {0:.0%}".format(prec))
    
    return {"Model Accuracy": acc, "Sensitivity":sens, "Specificity":spec, "Precision":prec}

tp, tn, fp, fn = ConfusionMatrix(y_predicted_bow, list(y_test))

print("(tp, tn,\n fp, fn)")
print(('({}, {},\n {}, {})\n\n'.format(tp, tn, fp, fn)))

dEval = EvalMetrics(tp, tn, fp, fn)


# When validating binary, use "metrics"  from scikit
from sklearn import metrics
print("\nsklearn.metrics\n---------------")
print("Accuracy (num correct / total):",metrics.accuracy_score(y_test, y_predicted_bow))
print("Sensitivity/Recall (true negs / all actual negs):",metrics.recall_score(y_test, y_predicted_bow))
print("Precision (true positives / all predicted positives):",metrics.precision_score(y_test, y_predicted_bow))

# 44, 98, 52, 60
###
# Try using TF-IDF
###
start = perf_counter()

# Create TF-IDF vectorizer
tfidf_vector = TfidfVectorizer(tokenizer = Tokenize, ngram_range=(1,1))

# Create the TF-IDF pipeline and fit the model
tfidf_pipe = Pipeline([("Vectorizer", tfidf_vector),
                 ("Classifier", model)])
tfidf_pipe.fit(X_train, y_train)


# Predict values using the trained model
y_predicted_tfidf = tfidf_pipe.predict(X_test)


# Measure Performance
stop = perf_counter()
print("Time to train and evaluate model (in seconds):", stop-start) 


tp, tn, fp, fn = ConfusionMatrix(y_predicted_tfidf, list(y_test))
print("(tp, tn,\n fp, fn)")
print(('({}, {},\n {}, {})\n\n'.format(tp, tn, fp, fn)))

dEval = EvalMetrics(tp, tn, fp, fn)
country_roads_lyrics ='''
    Almost heaven, West Virginia
    Blue Ridge Mountains, Shenandoah River
    Life is old there, older than the trees
    Younger than the mountains, growing like a breeze
    Country roads, take me home
    To the place I belong
    West Virginia, mountain mama
    Take me home, country roads
    All my memories gather 'round her
    Miner's lady, stranger to blue water
    Dark and dusty, painted on the sky
    Misty taste of moonshine, teardrop in my eye
    Country roads, take me home
    To the place I belong
    West Virginia, mountain mama
    Take me home, country roads
    I hear her voice, in the morning hour she calls me
    The radio reminds me of my home far away
    Driving down the road, I get a feeling
    That I should have been home yesterday, yesterday
    Country roads, take me home
    To the place I belong
    West Virginia, mountain mama
    Take me home, country roads
    Country roads, take me home
    To the place I belong
    West Virginia, mountain mama
    Take me home, country roads
    Take me home, down country roads
    Take me home, down country roads'''

country_roads = [country_roads_lyrics]

# Train the model
tfidf_pipe_full = Pipeline([("Vectorizer", tfidf_vector),
                 ("Classifier", model)])
tfidf_pipe_full.fit(X, y)

# Predict if Country Roads is a banger
country_roads_isBanger = tfidf_pipe_full.predict(country_roads)

if country_roads_isBanger: print("Computers don't lie... Country Roads is a banger!")
else: print("Computers don't lie... Country Roads IS NOT a banger!")