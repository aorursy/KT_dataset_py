import nltk

import numpy as np

import pandas as pd

import os 

import re

import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')



N_MC = 400

CUTOFF_PROB = 0.5
df = pd.read_csv("/kaggle/input/phish-reviews/Reviews.csv")

df.head()
# Regex to remove punctuation

def remove_punctuation(review):

    symbols="[’'`!)(&@#.,/'`~:;|\?]"

    return re.sub('\r',' ', re.sub('\n', ' ', re.sub(symbols,' ',review)))

# Remove newlines and convert all to lowercase

df.Reviews = df.Reviews.apply(remove_punctuation)

df.Reviews = df.Reviews.apply(lambda n: n.lower())
df.Reviews
# Tokenize Reviews

df.Reviews = df.Reviews.apply(nltk.word_tokenize)
# Filter Stop Words

stops = nltk.corpus.stopwords.words('english')

def filter_stop_words(tk_review):

    return [word for word in tk_review if word not in stops]
df.Reviews=df.Reviews.apply(filter_stop_words)
# POS-Tagging

df['Reviews2'] = df.Reviews.apply(nltk.pos_tag)
# Lemmatization

Lemming = nltk.stem.WordNetLemmatizer()



def penn2morphy(penntag):

    """ Converts Penn Treebank tags to WordNet. """

    morphy_tag = {'NN':'n', 'JJ':'a',

                  'VB':'v', 'RB':'r'}

    try:

        return morphy_tag[penntag[:2]]

    except:

        return 'n'



def lemmatize_words(tag_review):

    output = []

    for w in tag_review:

        try:

            lm = Lemming.lemmatize(w[0], pos=penn2morphy(w[1]))

        except Exception as e:

            lm = w

        output.append(lm)

        

    return output

df['Reviews2'] = df.Reviews2.apply(lemmatize_words)
df.head()
# Bin Scores and view Histogrm

def bin_score(row):

    return 1 if row['Score'] >= 4 else 0

df['BinnedScore'] = df.apply(bin_score, axis=1)

df.groupby('BinnedScore').size().plot(kind='bar')
# Extract N most common words

def n_most_common(reviews, n=100):

    words = []

    sentences = reviews.values

    

    # Iterate through sentences to extract all unique words

    for s in sentences:

        for w in s:

            if w not in words:

                words.append(w)

    

    # Generate a counter for each word

    counts = {w: 0 for w in words}

    

    # Iterate through all words, incrementing count

    for sent in sentences:

        for word in sent:

            counts[word] += 1

    

    # Generate list of 

    words = sorted([(w, counts[w]) for w in words], key= lambda x: x[1], reverse=True)

    return words[0:n]

    
MC = n_most_common(df.Reviews2,N_MC)
# Create Categorical Variables

MC = [w[0] for w in MC]

def create_categorical_column(dframe, word):

    def hasword(row):

        if word in row['Reviews2']:

            return 1

        else:

            return 0

    dframe[word] = dframe.apply(lambda row: hasword(row), axis=1)
DF = df.drop(columns=['Unnamed: 0', 'Reviews', 'Score'])

DF.head()

DF.to_csv('CleanedDF1.csv')
DF.head()
#One-Hot Encoding

def encode_word(row, word):

    return 1 if word in row['Reviews2'] else 0

lenmc = len(MC)

for i, word in enumerate(MC):

    DF[str(word)] = DF.apply(lambda row: encode_word(row, word), axis=1)

    print(f'{i}/{lenmc}')

    
from sklearn.model_selection import train_test_split





train, test = train_test_split(DF, test_size=0.2)



train_y = train.pop('BinnedScore')

test_y = test.pop('BinnedScore')
train.drop('Reviews2', inplace=True, axis=1)

test.drop('Reviews2', inplace=True, axis=1)
from sklearn.naive_bayes import ComplementNB, GaussianNB

model = ComplementNB(alpha=1)

model.fit(train, train_y)
import random

def pred_cat(prob, CUTOFF_PROB=CUTOFF_PROB):

    if prob > CUTOFF_PROB:

        return '1'

    elif prob < CUTOFF_PROB:

        return '0'

    else:

        return random.choice([0, 1])
# Make Predictions!

probs = [round(p[1], 4) for p in model.predict_proba(test)]

predictions = [pred_cat(p) for p in probs]

observed = [v for v in test_y.values]

benchmark_predictions = [1 for i in test_y.values]



results = pd.DataFrame([predictions, observed, benchmark_predictions, probs]).transpose()

results.rename({0:'Predicted', 1:'Observed', 2:'Benchmark_Predictions', 3:'Probabilities'}, axis=1, inplace=True)



results = results.apply(lambda x: x.apply(str))

results
def model_right(row):

    return 1 if row['Predicted'] == row['Observed'] else 0



def benchmark_right(row):

    return 1 if row['Benchmark_Predictions'] == row['Observed'] else 0

results['model_right'] = results.apply(model_right, axis=1)

results['benchmark_right'] = results.apply(benchmark_right, axis=1)
results
from sklearn.metrics import roc_auc_score

AUCstat = roc_auc_score(results['Observed'].apply(float).values.flatten(), results['Probabilities'].apply(float).values.flatten())
print(f'model error: {1 - results["model_right"].sum() / len(results)}')

print(f'AUC: {AUCstat}')

print(f'benchmark error: {1 - results["benchmark_right"].sum() / len(results)}')
Positives = results[results['Observed'] == '1']

Negatives = results[results['Observed'] == '0']



sensitivity = len(Positives[Positives['Predicted']=='1']) / len(Positives)

specificity = len(Negatives[Negatives['Predicted']=='0']) / len(Negatives)



print(f'Sensitivty: {round(sensitivity*100, 2)}%')

print(f'Specificity: {round(specificity*100, 2)}%')
results
def ROC(rdf):

    

    df1 = pd.concat([rdf['Probabilities'], rdf['Observed']], axis=1)



    ss = []

    sc = []    

    

    def populate_ss(ctf):

        

        df1['TempPred'] = df1['Probabilities'].apply(lambda p: pred_cat(float(p), CUTOFF_PROB=float(ctf)))

        

        Positives = df1[df1['Observed'] == '1']

        Negatives = df1[df1['Observed'] == '0']

        

        sensi = len(Positives[Positives['TempPred'] == '1']) / len(Positives)

        speci = len(Negatives[Negatives['TempPred'] == '0']) / len(Negatives)

        

        ss.append(sensi)

        sc.append(1-speci)



    cutoffs = np.arange(0, 1, 0.005)

    for i in cutoffs:

        populate_ss(i)



    plt.title("ROC Chart")

    plt.xlabel('1-Specificity')

    plt.ylabel('Sensitivity')

    

    plt.plot([0, 0, 1], [0, 1, 1], c='g', label='Ideal')

    plt.plot(sc, ss, c='b', label='Observed')

    plt.plot([0, 1], [0, 1], c='r', label='Benchmark')

    

    plt.legend()
ROC(results)