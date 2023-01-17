import pandas as pd
import glob

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model
from keras.layers import *
#Converting all input data to csv files
'''allFiles = glob.glob("movie_reviews/train/pos/*")

train=pd.DataFrame()

for file_ in allFiles:
    with open(file_, 'r') as myfile:
        data=myfile.read().replace('<br />', '').replace('\\', '')
        train = train.append({'text':data} , ignore_index=True)
        
train_pos = train
train.to_csv("train_pos.csv", sep=',',index=False)'''

'''allFiles = glob.glob("movie_reviews/train/neg/*")

train_neg=pd.DataFrame()

for file_ in allFiles:
    with open(file_, 'r') as myfile:
        data=myfile.read().replace('<br />', '').replace('\\', '')
        train_neg = train_neg.append({'text':data} , ignore_index=True)
        
train_neg.to_csv("train_neg.csv", sep=',',index=False)'''

'''#empty folders "pos" and "neg" were deleted from folder "test"

testFiles = glob.glob("movie_reviews/test/*")

test=pd.DataFrame()

for file_ in testFiles:
    with open(file_, 'r') as myfile:
        data=myfile.read().replace('<br />', '').replace('\\', '')
        test = test.append({'text':data} , ignore_index=True)
        
test.to_csv("test.csv", sep=',',index=False)'''

'''unsupFiles = glob.glob("movie_reviews/unsup/*")

unsup_in=pd.DataFrame()

for file_ in unsupFiles:
    with open(file_, 'r') as myfile:
        data=myfile.read().replace('<br />', '').replace('\\', '')
        unsup_in = unsup_in.append({'text':data} , ignore_index=True)
        
unsup_in.to_csv("unsup_in.csv", sep=',',index=False)'''
pd.set_option('display.max_colwidth', -1)

train_pos = pd.read_csv("../input/positive-and-negative-movies-reviews/train_pos.csv", sep=',')
train_pos['text_length'] = train_pos['text'].apply(len)
train_pos['sentiment'] = 1

train_neg = pd.read_csv("../input/positive-and-negative-movies-reviews/train_neg.csv", sep=',')
train_neg['text_length'] = train_neg['text'].apply(len)
train_neg['sentiment'] = 0

train = train_pos.append(train_neg)

test = pd.read_csv("../input/movies-sentiment-test/test.csv", sep=',')
unsup = pd.read_csv("../input/positive-and-negative-movies-reviews/unsup_in.csv", sep=',')
print(train['text_length'].describe())

#Check if there are any null in dataframe
print(train.isnull().values.any())
print(test.isnull().values.any())
train_df, val_df = train_test_split(train[['text','sentiment']], test_size=0.1, random_state=2018)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 1500 # max number of words in a review to use. 75% of all questions < 1500

train_X = train_df['text'].values
train_y = train_df['sentiment'].values

val_X = val_df['text'].values
val_y = val_df['sentiment'].values

test_X = test['text'].values

unsup_X = unsup['text'].values
## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)
unsup_X = tokenizer.texts_to_sequences(unsup_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)
unsup_X = pad_sequences(unsup_X, maxlen=maxlen)
model1=Sequential() 
model1.add(Embedding(max_features, embed_size, input_length=maxlen))
model1.add(Dropout(0.5))
model1.add(Flatten())
model1.add(Dense(256, activation='relu')) 
model1.add(Dropout(0.5))
model1.add(Dense(64, activation='relu')) 
model1.add(Dropout(0.5))
model1.add(Dense(16, activation='relu')) 
model1.add(Dropout(0.5))
model1.add(Dense(1,activation='sigmoid')) 
model1.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model1.fit(train_X,train_y, batch_size=512, epochs=10, validation_data=[val_X, val_y])
pred_val_y = model1.predict([val_X], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 1.001, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh_1 = thresholds[0][0]
print("Best threshold: ", best_thresh_1)

inp = Input(shape=(maxlen, ))
embed = Embedding(max_features, embed_size)(inp)
embed = Dropout(0.5)(embed)
filter_sizes = [1,2,3,5]
#filter_sizes = [5]
num_filters = 64

conv_0 = Conv1D(num_filters, filter_sizes[0], padding='valid', kernel_initializer='normal', activation='relu')(embed)
conv_1 = Conv1D(num_filters, filter_sizes[1], padding='valid', kernel_initializer='normal', activation='relu')(embed)
conv_2 = Conv1D(num_filters, filter_sizes[2], padding='valid', kernel_initializer='normal', activation='relu')(embed)
conv_3 = Conv1D(num_filters, filter_sizes[3], padding='valid', kernel_initializer='normal', activation='relu')(embed)

maxpool_0 = MaxPool1D(pool_size=(maxlen - filter_sizes[0] + 1), strides=(1), padding='valid')(conv_0)
maxpool_1 = MaxPool1D(pool_size=(maxlen - filter_sizes[1] + 1), strides=(1), padding='valid')(conv_1)
maxpool_2 = MaxPool1D(pool_size=(maxlen - filter_sizes[2] + 1), strides=(1), padding='valid')(conv_2)
maxpool_3 = MaxPool1D(pool_size=(maxlen - filter_sizes[3] + 1), strides=(1), padding='valid')(conv_3)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])

x = Flatten()(concatenated_tensor)
#x = Flatten()(maxpool_0)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.5)(x)
outp = Dense(1, activation="sigmoid")(x)
model2 = Model(inputs=inp, outputs=outp)
model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])    
model2.fit(train_X,train_y, batch_size=512, epochs=6, validation_data=[val_X, val_y])
pred_val_y = model2.predict([val_X], batch_size=512, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 1.001, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh_2 = thresholds[0][0]
print("Best threshold: ", best_thresh_2)

model3 = Sequential()
model3.add(Embedding(max_features, embed_size, input_length=maxlen))
model3.add(Dropout(0.5))
model3.add(LSTM(128, return_sequences=True))
model3.add(LSTM(128))
model3.add(Dropout(0.5))
model3.add(Dense(32))
model3.add(Dropout(0.5))
model3.add(Dense(1,activation='sigmoid'))
model3.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model3.fit(train_X,train_y, batch_size=512, epochs=10, validation_data=[val_X, val_y])
pred_val_y = model3.predict([val_X], batch_size=512, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 1.001, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh_3 = thresholds[0][0]
print("Best threshold: ", best_thresh_3)

pred_test = model3.predict([test_X], batch_size=512, verbose=1)
pred_unsup = model3.predict([unsup_X], batch_size=512, verbose=1)

#applying the best threshold
pred_test = pred_test > best_thresh_3
pred_unsup = pred_unsup > best_thresh_3

test['sentiment'] = pred_test
unsup['sentiment'] = pred_unsup
#write the output
test.to_csv("test_results.csv", sep=',',index=False)
unsup.to_csv("unsup_results.csv", sep=',',index=False)
