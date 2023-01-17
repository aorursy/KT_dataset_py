import pandas as pd #The good stuff
import numpy as np
train = pd.read_csv('../input/imdb_train.csv') #Load training data
train = train[(train.label == 'pos') | (train.label == 'neg')] #Remove everything that is not binary
train.head()
reviews = train.review.values
label = train.label == 'pos' #One hot encode
from keras.preprocessing.text import Tokenizer
tk = Tokenizer(num_words=10000) 
tk.fit_on_texts(reviews) #Create token dictionary
sq = tk.texts_to_sequences(reviews) #Convert to sequence
from keras.preprocessing.sequence import pad_sequences
sq = pad_sequences(sq,maxlen=100) #Ensures all sequences have the same length
from keras.models import Sequential, Model
from keras.layers import Embedding,CuDNNLSTM, Dense, Input
model = Sequential()
model.add(Embedding(input_dim=10000,output_dim=32)) #We make no assumptions about input length
model.add(CuDNNLSTM(32))
model.add(Dense(1,activation='sigmoid'))
inp = Input(shape=(None,)) #We make no assumptions about input length
x = Embedding(input_dim=10000,output_dim=32)(inp)
x = CuDNNLSTM(32)(x)
out = Dense(1,activation='sigmoid')(x)
model = Model(inp,out)
model.compile('adam', 'binary_crossentropy', metrics=['acc'])
model.fit(sq,label)
test = pd.read_csv('../input/sample_submission.csv')
test.head()
test_revs = test.review.values
test_sq = tk.texts_to_sequences(test_revs) #Use the same token dictionary as created above
test_sq = pad_sequences(test_sq,maxlen=100) #Ensure same length
preds = (model.predict(test_sq) >= 0.5) #Make predictions
preds = preds.flatten() #Flatten vector for pandas
test.Category = 'neg' #Default value
test.loc[preds==1,'Category'] = 'pos'
test.head()
test.drop('review',axis=1).to_csv('submission.csv',index=False)
