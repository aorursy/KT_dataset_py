import sqlite3
import pandas
import re

# read the data
con = sqlite3.connect('../input/database.sqlite')
dataset = pandas.read_sql_query("SELECT Score, Text FROM Reviews;", con)
texts = dataset['Text']
scores = list(dataset['Score'])

# helper function to strip caps and punctuation
def strip(s):
    return re.sub(r'[^\w\s]','',s).lower()

# this is where the stemming happens (works better without)
#from nltk.stem.porter import PorterStemmer
#stemmer = PorterStemmer()
#print("Stemming...")
#texts = [' '.join([stemmer.stem(w) for w in s.split(' ')]) for s in texts]

# further cleanup
texts = [strip(sentence) for sentence in texts]

# now to vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
features = vectorizer.fit_transform(texts)

# create training and testing data sets, starting with the scores
numReviews = len(scores)
trainingScores = scores[0:int(numReviews*0.8)]
testingScores = scores[int(numReviews*0.8):]

# addPredictions will reduce feature dimensionality to k,
# & then build a predictive model using those features
def addPredictions(k):
    global con
    c = con.cursor()
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.linear_model import LogisticRegression
    # reduce dimensions
    sel = SelectKBest(chi2,k=k)
    kFeatures = sel.fit_transform(features,scores)
    # split reduced review data into training and testing sets
    trainingFeatures = kFeatures[0:int(numReviews*0.8)]
    testingFeatures = kFeatures[int(numReviews*0.8):]
    # fit the prediction model
    model = LogisticRegression(C=1e5)
    model.fit(trainingFeatures,trainingScores)
    # add the predictions to the database
    try:
        c.execute("alter table Reviews add column k{} integer;".format(str(k)))
    except:
        pass
    for n in xrange(0,len(scores)):
        k = str(k)
        p = str(model.predict(kFeatures[n])[0])
        i = str(n+1)
        c.execute("update Reviews set k{}={} where Id={}".format(k,p,i))
    con.commit()
    
#addPredictions(10)
#addPredictions(100)
#addPredictions(1000)
#addPredictions(10000)
#addPredictions(100000)
