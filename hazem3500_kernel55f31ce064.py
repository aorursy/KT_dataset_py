import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from gensim.models import Word2Vec
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import re
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
def sentanceEmbed(sentance, wordsMatrix):
    sentanceFeatureMatrix = np.zeros(300)
    for word in sentance:
        sentanceFeatureMatrix += wordsMatrix[word]
    sentanceFeatureMatrix /= len(sentance)
    return sentanceFeatureMatrix

def filterTweet(sentance):
    txt=re.sub(r'@[A-Z0-9a-z_:]+','',sentance)
    txt=re.sub(r'^[RT]+','',txt)
    txt = re.sub('https?://[A-Za-z0-9./]+','',txt)
    txt=re.sub("[^a-zA-Z]", " ",txt)
    txt=txt.lower()
    return txt

def mapLabelToNumericValue(s):
    if(s=="positive"):
        return 1
    if(s=="negative"):
        return 2
    if(s=="neutral"):
        return 3

dataset = pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")


dataFrame = pd.DataFrame()
dataFrame['label'] = dataset['airline_sentiment'].apply(lambda sentance : mapLabelToNumericValue(sentance))
dataFrame['tweet'] = dataset['text'].apply(lambda sentance : word_tokenize(filterTweet(sentance)))

model = Word2Vec(dataFrame['tweet'], min_count=1,window=5,size=300)

dataFrame['tweet'] = dataFrame['tweet'].apply(lambda sentance : sentanceEmbed(sentance, model.wv))


cols = ['F{0}'.format(i) for i in range(1, 301)]
f = pd.DataFrame(columns=cols)

for row in dataFrame['tweet']:
    data_to_append = {}
    for i in range(len(f.columns)):
        data_to_append[f.columns[i]] = row[i]
    f = f.append(data_to_append, ignore_index = True)

f['label'] = dataFrame['label']

positive = f.loc[f['label'] == 1]
neutral = f.loc[f['label'] == 3].sample(80)
negative = f.loc[f['label'] == 2].sample(80)

positiveTrain = positive.head(80)
neutralTrain = neutral.head(80)
negativeTrain = negative.head(80)

positiveTest = positive.tail(80)
neutralTest = neutral.tail(80)
negativeTest = negative.tail(80)

trainingData = pd.concat([positiveTrain, neutralTrain, negativeTrain], ignore_index=True, sort=False)
testingData = pd.concat([positiveTest, neutralTest, negativeTest], ignore_index=True, sort=False)

xTrain = trainingData[cols]
yTrain = trainingData['label']
xTest = testingData[cols]
yTest = testingData['label']
classifier = svm.SVC()
classifier.fit(xTrain, yTrain)
predicted = classifier.predict(xTest)
print(accuracy_score(yTest, predicted))
print(classifier.predict(sentanceEmbed(word_tokenize(filterTweet("what said at travel")), model)))


