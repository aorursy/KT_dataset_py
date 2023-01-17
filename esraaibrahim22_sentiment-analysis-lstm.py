#import os
#os.remove("/kaggle/working/sntimentModel.sav")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from string import punctuation
import re
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer

from sklearn.utils import resample
from sklearn.utils import shuffle





data = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')
review=data['text']
label=data['airline_sentiment']

print("after reading :",len(review) , len(label))

df_class_0 = data[data['airline_sentiment'] == 'negative']
df_class_1 = data[data['airline_sentiment'] == 'positive']
df_class_2 = data[data['airline_sentiment'] == 'neutral']
target_count = data.airline_sentiment.value_counts()
    
print(target_count[0] , target_count[1] , target_count[2])
"""
df_class_1_over = df_class_1.sample(target_count[0], replace=True)
df_class_2_over = df_class_2.sample(target_count[0], replace=True)


df_test_over_total = pd.concat([df_class_0,df_class_1_over, df_class_2_over ], axis=0)

print('Random over-sampling:')
print(df_test_over_total.airline_sentiment.value_counts())

review=df_test_over_total.text
label=df_test_over_total.airline_sentiment

print("after overSampling:",len(review) , len(label))
"""
train = pd.concat([df_class_0.sample(frac=0.8,random_state=200),
         df_class_1.sample(frac=0.8,random_state=200), df_class_2.sample(frac=0.8,random_state=200)])
test = pd.concat([df_class_0.drop(df_class_0.sample(frac=0.8,random_state=200).index),
        df_class_1.drop(df_class_1.sample(frac=0.8,random_state=200).index),
                  df_class_2.drop(df_class_2.sample(frac=0.8,random_state=200).index)])

train = shuffle(train)
test = shuffle(test)

print('positive data in training:',(train.airline_sentiment == 'positive').sum())
print('negative data in training:',(train.airline_sentiment == 'negative').sum())
print('neutral data in training:',(train.airline_sentiment == 'neutral').sum())
print('positive data in test:',(test.airline_sentiment == 'positive').sum())
print('negative data in test:',(test.airline_sentiment == 'negative').sum())
print('neutral data in test:',(test.airline_sentiment == 'neutral').sum())
neg = train[train['airline_sentiment'] == 'negative']
pos = train[train['airline_sentiment'] == 'positive']
neu = train[train['airline_sentiment'] == 'neutral']

pos_upsampled = resample(pos, 
                                 replace=True,     # sample with replacement
                                 n_samples= neg.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
neu_upsampled = resample(neu, 
                                 replace=True,     # sample with replacement
                                 n_samples= neg.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
data_upsampled = pd.concat([neg, pos_upsampled ,neu_upsampled ])
print("After upsampling\n",data_upsampled.airline_sentiment.value_counts(),sep = "")
review=data_upsampled.text
label=data_upsampled.airline_sentiment

testReview=test.text
testLabel=test.airline_sentiment

model = KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True )

word_vectors = model.wv

def sentenceEmbedding(tokens , typee="avg"):
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
        if typee== "sum":
            result[i]=res
        else:
            result[i]=(res/(len(tokens)))
        res=0
         
    return result
"""
def mapSentiment(airlineSentiment):
    if airlineSentiment == 'positive':
        return 1
    elif airlineSentiment == 'negative' :
        return 0
    else:
        return 2
        """
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


label = pd.get_dummies(data_upsampled.airline_sentiment).values
testLabel = pd.get_dummies(test.airline_sentiment).values
print (len(label) , label)


review=cleanString(review)
testReview=cleanString(testReview)

t = input("choose sum ||  avg for sentence embedding:")

testVectors=[]
featureVectors=[]
for r in review:
    sentence=r.split()
    featureVectors.append(sentenceEmbedding(sentence,t))
for r in testReview:
    sentence=r.split()
    testVectors.append(sentenceEmbedding(sentence,t))


print("last:",len(featureVectors) , len(label))

featureVectors, xVal, label, yVal = train_test_split(featureVectors, label, test_size=0.15, random_state=42)


from keras.models import Sequential
from keras.layers import LSTM , Dense , Dropout, Activation,SpatialDropout1D
from keras.utils import to_categorical
from keras.optimizers import SGD

xTrain=np.array(featureVectors)
yTrain=np.array(label)
xTest=np.array(testVectors)
yTest=np.array(testLabel)
xVal=np.array(xVal)
yVal=np.array(yVal)
""""
xTrain=np.array(X_train)
yTrain=np.array(y_train)
xTest=np.array(X_test)
yTest=np.array(y_test)"""
print("XTRAIN_SHAPE:" , xTrain.shape , "YTRAIN_SHAPE:", yTrain.shape)

yTrain=np.reshape(yTrain , ( yTrain.shape[0],1,3 ))
yTest=np.reshape(yTest , (yTest.shape[0],1,3 ))
yVal=np.reshape(yVal , (yVal.shape[0] ,1, 3))

xTrain=np.reshape(xTrain , (xTrain.shape[0] ,1, 300))
xTest=np.reshape(xTest , (xTest.shape[0] ,1, 300))
xVal=np.reshape(xVal , (xVal.shape[0] ,1, 300))

print("XTRAIN_RESHAPE:" , xTrain.shape , "YTRAIN_RESHAPE:", yTrain.shape)


input_length = None
input_dim = 300
c_model=Sequential()



c_model.add(LSTM(265, dropout=0.4, recurrent_dropout=0.4,input_dim = input_dim
                 , input_length = input_length,return_sequences=True ) )
#c_model.add(LSTM(50))

c_model.add(Dense(3,activation='softmax'))
c_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(c_model.summary())

history=c_model.fit(xTrain, yTrain,nb_epoch=10 ,  batch_size=128, verbose = 1 , validation_data=(xVal, yVal) )
# plot train and validation loss
"""
from matplotlib import pyplot

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()"""
y=c_model.predict_classes(xTest,batch_size = 128)
print (y)
loss , accur=c_model.evaluate(xTest , yTest)
print("loss:", loss , "\nacc:", accur)
def lstmPrediction(review ,choice ):
    smbls=dict()
    smbls[0]="Negative"
    smbls[1]="Neutral"
    smbls[2]="Positive"
    r=cleanString([review])
    review=sentenceEmbedding(r[0].split() , choice)
    data=np.array([review])
    data=np.reshape(data,(1,data.shape[0],300))
    ps=c_model.predict(data)
    print(ps)
    return smbls[np.argmax(ps)]
review=input("Enter your review: ")
choice=input("Enter sum || avg: ")
print(lstmPrediction(review ,choice ))
