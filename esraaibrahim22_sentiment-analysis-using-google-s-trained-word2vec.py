#import os
#os.remove("/kaggle/working/sntimentModel.sav")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from string import punctuation
import pickle , re
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
from sklearn.utils import shuffle



data = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')
review=data['text']
label=data['airline_sentiment']

df_class_0 = data[data['airline_sentiment'] == 'negative']
df_class_1 = data[data['airline_sentiment'] == 'positive']
df_class_2 = data[data['airline_sentiment'] == 'neutral']
target_count = data.airline_sentiment.value_counts()
    
print(target_count[0] , target_count[1] , target_count[2])

df_class_1_over = df_class_1.sample(target_count[0], replace=True)
df_class_2_over = df_class_2.sample(target_count[0], replace=True)


df_test_over_total = pd.concat([df_class_0,df_class_1_over, df_class_2_over ], axis=0)

print('Random over-sampling:')
print(df_test_over_total.airline_sentiment.value_counts())

review=df_test_over_total.text
label=df_test_over_total.airline_sentiment





model = KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True )

word_vectors = model.wv

def sentenceEmbedding(tokens):
    vectors=[]
    for word in tokens:
        if word not in word_vectors.vocab:
            vectors.append([0]*300) #size of feature vector
        else:
            vectors.append(model[word])
    
    result=[0] * len(vectors[0])
    res=0
    for i in range(len(vectors[0])):
         for vec in vectors:
                res+=vec[i]
         result[i]=(res/(len(tokens)))
         res=0
         
    return result

def mapSentiment(airlineSentiment):
    if airlineSentiment == 'positive':
        return 1
    elif airlineSentiment == 'negative' :
        return 0
    else:
        return 2
    
wordnet_lemmatizer = WordNetLemmatizer()
def cleanString(sentences):
    result=[]
    for sen in sentences:
        s=""
        r=""
        s+=(sen.lower()+' ')
        s = re.sub("(@\w* )", ' ', s)
        s = re.sub("\\bhttps://(.*) \\b",' ',s) 
        s = re.sub("[^a-z0-9\ ]+", ' ', s)
        s = re.sub(' \d+', ' ', s)
        s = re.sub(" +",' ',s)
        tokens=s.split()
        for w in tokens:
             r+=wordnet_lemmatizer.lemmatize(w ,pos="v")+" "
        result.append(r)
    return result


label = [ mapSentiment(x) for x in label]
review=cleanString(review)

featureVectors=[]
for r in review:
    sentence=r.split()
    featureVectors.append(sentenceEmbedding(sentence))


X_train, X_test, y_train, y_test = train_test_split(featureVectors, label, test_size=0.2, random_state=42)
classification_model = LogisticRegression(solver='newton-cg', C=1e7)
classification_model.fit(X_train, y_train)
yPrediction = classification_model.predict(X_test)
print("prediction:" ,yPrediction)
acc =accuracy_score(y_test, yPrediction)
print("Accuracy - LogisticRegression:",acc*100)


filename = 'sentimentModel.sav'
pickle.dump(classification_model, open(filename, 'wb'))


"""cModel2=RandomForestClassifier(n_estimators=200)
cModel2.fit(X_train, y_train)
yPrediction2 = cModel2.predict(X_test)
print("prediction:" ,yPrediction2)
acc2 =accuracy_score(y_test, yPrediction2)
print("Accuracy - RandomForestClassifier:",acc2 *100)
"""
"""
user input
"""
while True:
    userInput=input("Enter your review to test model:")
    userInput=cleanString([userInput])
    userTokens=userInput[0].split()
    
    print(userTokens)
    
    fVec=sentenceEmbedding(userTokens)
    y=classification_model.predict([fVec])
    #y2=cModel2.predict([fVec])
    
    if(y[0]==1):
        print("LogisticRegression:  Positive")
    elif(y[0]==0):
        print("LogisticRegression:  Negative")
    else:
        print("LogisticRegression:  Neutral")
    """  
    if(y2[0]==1):
        print("RandomForestClassifier:  Positive")
    elif(y2[0]==0):
        print("RandomForestClassifier:  Negative")
    else:
        print("RandomForestClassifier:  Neutral")"""
