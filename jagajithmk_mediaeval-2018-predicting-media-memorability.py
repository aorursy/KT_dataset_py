def Get_score(Y_pred,Y_true):

    '''Calculate the Spearmann"s correlation coefficient'''

    Y_pred = np.squeeze(Y_pred)

    Y_true = np.squeeze(Y_true)

    if Y_pred.shape != Y_true.shape:

        print('Input shapes don\'t match!')

    else:

        if len(Y_pred.shape) == 1:

            Res = pd.DataFrame({'Y_true':Y_true,'Y_pred':Y_pred})

            score_mat = Res[['Y_true','Y_pred']].corr(method='spearman',min_periods=1)

            print('The Spearman\'s correlation coefficient is: %.3f' % score_mat.iloc[1][0])

        else:

            for ii in range(Y_pred.shape[1]):

                Get_score(Y_pred[:,ii],Y_true[:,ii])
!pip install pyprind

!pip install keras

!pip install tensorflow
import pandas as pd

from keras import Sequential

from keras import layers

from keras import regularizers

import numpy as np

from string import punctuation

import pyprind

from collections import Counter

from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import os
# for reproducability

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(1)
# load labels and captions

def read_caps(fname):

    """Load the captions into a dataframe"""

    vn = []

    cap = []

    df = pd.DataFrame();

    with open(fname) as f:

        for line in f:

            pairs = line.split()

            vn.append(pairs[0])

            cap.append(pairs[1])

        df['video']=vn

        df['caption']=cap

    return df



# load the captions

df_cap=read_caps('../input/data/data/dev-set_video-captions.txt')



# load the ground truth values

labels=pd.read_csv('../input/data/data/dev-set_ground-truth.csv')
labels.head()
from nltk.stem import WordNetLemmatizer

from nltk import tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

import re
df = df_cap.copy()

import re

def strip_character(dataCol):

    r = re.compile(r'[^a-zA-Z]')

    return r.sub(' ', str(dataCol))



df['caption'] = df['caption'].apply(strip_character)
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

import nltk

nltk.download('wordnet')
stop = stopwords.words('english') 
df['caption'] = df['caption'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

df['caption'].head()
df['caption'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in df['caption']]

df['caption'].head()
counts = Counter()

for i, cap in enumerate(df['caption']):

    counts.update(cap.split())
df.caption.values
vect = TfidfVectorizer(ngram_range = (1,4)).fit(df.caption)

vect_transformed_X_train = vect.transform(df.caption)

len_token = len(vect.get_feature_names())
len_token
# build the word index

len_token = len(counts)

tokenizer = Tokenizer(num_words=len_token)
tokenizer.fit_on_texts(list(vect.get_feature_names())) #fit a list of captions to the tokenizer

#the tokenizer vectorizes a text corpus, by turning each text into either a sequence of integers 
one_hot_res = tokenizer.texts_to_matrix(list(df.caption.values),mode='binary')

#sequences = tokenizer.texts_to_sequences(list(df.caption.values))
len(one_hot_res)
one_hot_res.shape
Y = labels[['short-term_memorability','long-term_memorability']].values

X = one_hot_res;

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)



# add dropout

# add regularizers



model = Sequential()

model.add(layers.Dense(200,activation='relu',kernel_regularizer=None,input_shape=(len_token,)))

model.add(layers.Dropout(0.1))

model.add(layers.Dense(2,activation='sigmoid'))





          

# compile the model 

model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])



# training the model 

history = model.fit(X_train,Y_train,epochs=20,validation_data=(X_test,Y_test))



# visualizing the model

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1,len(loss)+1)



plt.plot(epochs,loss,'bo',label='Training loss')

plt.plot(epochs,val_loss,'b',label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



plt.figure()

acc = history.history['acc']

val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Acc')

plt.legend()

plt.show()
predictions = model.predict(X_test)

print(predictions.shape)
Get_score(predictions, Y_test)
model.save('CaptionPrdiction_NN.h5')  # creates a HDF5 file 'my_model.h5'
from sklearn.ensemble import RandomForestRegressor
rf3 = RandomForestRegressor(n_estimators = 10, random_state = 0).fit(X_train, Y_train);
y_predict = rf3.predict(X_test)

Get_score(y_predict, Y_test)
from sklearn.linear_model import LinearRegression

from sklearn.datasets import make_regression

modelLR = LinearRegression().fit(X_train, Y_train)
ynew = modelLR.predict(X_test)

Get_score(y_predict, Y_test)
from sklearn.svm import SVR

Y_short = labels[['short-term_memorability']].values

X = one_hot_res;

X_train, X_test, Y_train_short, Y_test_short = train_test_split(X,Y_short, test_size=0.2, random_state=42)

modelSVR_short = SVR(C=100).fit(X_train,Y_train_short)
predictionsSVR_short = modelSVR_short.predict(X_test)

Get_score(predictionsSVR_short,Y_test_short)
Y_long = labels[['long-term_memorability']].values

X_train, X_test,Y_train_long, Y_test_long = train_test_split(X,Y_long, test_size=0.2, random_state=42)
Y_test_long.shape

modelSVR_long = SVR(C=100).fit(X_train,Y_train_long)
predictionsSVR_long = modelSVR_long.predict(X_test)

Get_score(predictionsSVR_long,Y_test_long)
from sklearn.tree import DecisionTreeRegressor
regr = DecisionTreeRegressor(max_depth=10)

regr.fit(X_train, Y_train)

pred_test_dtr = regr.predict(X_test)
Get_score(pred_test_dtr, Y_test)
def read_C3D(fname):

    """Scan vectors from file"""

    with open(fname) as f:

        for line in f:

            C3D =[float(item) for item in line.split()] # convert to float type, using default separator

    return C3D



def vname2ID(vnames):

    """Parse video digital id from its name

    vnames: a list contains file names"""

    vid = [ os.path.splitext(vn)[0]+'.webm' for vn in vnames]

    return vid
C3D_Feat_path = '../input/data/data/dev-set_features/'

# Load video related features first

# it helps with the organization of the video names

vid = labels.video.values



C3D_Features = pd.DataFrame({'video': vid,

                   'C3D': [read_C3D(C3D_Feat_path+'C3D'+'/'+os.path.splitext(item)[0]+'.txt') for item in vid],

                       })
C3D_X = np.stack(C3D_Features['C3D'].values)

C3D_Y = labels[['short-term_memorability','long-term_memorability']].values



C3D_X_train, C3D_X_test, C3D_Y_train, C3D_Y_test = train_test_split(C3D_X,C3D_Y, test_size=0.2, random_state=42)
C3D_model = Sequential()

C3D_model.add(layers.Dense(200,activation='relu',kernel_regularizer=None,input_shape=(C3D_X.shape[1],)))

C3D_model.add(layers.Dropout(0.1))

C3D_model.add(layers.Dense(2,activation='sigmoid'))

C3D_model.compile(optimizer='rmsprop',loss=['mae'])

history=C3D_model.fit(x=C3D_X_train,y=C3D_Y_train,batch_size=50,epochs=20,validation_split=0.2,shuffle=True,verbose=True)

C3D_Y_pred = C3D_model.predict(C3D_X_test)

Get_score(C3D_Y_pred,C3D_Y_test)
from sklearn.ensemble import RandomForestRegressor

C3D_clf = RandomForestRegressor()

C3D_clf.fit(C3D_X_train,C3D_Y_train)

pred_test_rfr = C3D_clf.predict(C3D_X_test)

Get_score(pred_test_rfr, C3D_Y_test)
def read_HMP(fname):

    """Scan HMP(Histogram of Motion Patterns) features from file"""

    with open(fname) as f:

        for line in f:

            pairs=line.split()

            HMP_temp = { int(p.split(':')[0]) : float(p.split(':')[1]) for p in pairs}

    # there are 6075 bins, fill zeros

    HMP = np.zeros(6075)

    for idx in HMP_temp.keys():

        HMP[idx-1] = HMP_temp[idx]            

    return HMP
HMP_Feat_path = '../input/data/data/dev-set_features/'

# Load video related features first

# it helps with the organization of the video names

vid = labels.video.values

HMP_Features = pd.DataFrame({'video': vid,

                   'HMP': [read_HMP(HMP_Feat_path+'HMP'+'/'+os.path.splitext(item)[0]+'.txt') for item in vid],

                       })
HMP_X = np.stack(HMP_Features['HMP'].values)

HMP_Y = labels[['short-term_memorability','long-term_memorability']].values

HMP_X_train, HMP_X_test, HMP_Y_train, HMP_Y_test = train_test_split(HMP_X,HMP_Y, test_size=0.2, random_state=42)
HMP_model = Sequential()

HMP_model.add(layers.Dense(200,activation='relu',kernel_regularizer=None,input_shape=(HMP_X.shape[1],)))

HMP_model.add(layers.Dropout(0.1))

HMP_model.add(layers.Dense(2,activation='sigmoid'))

HMP_model.compile(optimizer='rmsprop',loss=['mae'])

history=HMP_model.fit(x=HMP_X_train,y=HMP_Y_train,batch_size=50,epochs=20,validation_split=0.2,shuffle=True,verbose=True)

HMP_Y_pred = HMP_model.predict(HMP_X_test)

Get_score(HMP_Y_pred,HMP_Y_test)
from sklearn.ensemble import RandomForestRegressor

HMP_clf = RandomForestRegressor()

HMP_clf.fit(HMP_X_train,HMP_Y_train)

pred_test_rfr = HMP_clf.predict(HMP_X_test)

Get_score(pred_test_rfr, HMP_Y_test)
# load the captions

#cap_path = '/media/win/Users/ecelab-adm/Desktop/DataSet_me18me/me18me-devset/dev-set/dev-set_video-captions.txt'

cap_path = '../input/data/data/test-set_video-captions.txt'

df_test=read_caps(cap_path)



# load the ground truth values

test_ground_truth=pd.read_csv('../input/data/data/test-set_ground-truth.csv')



df_test['caption'] = df_test['caption'].apply(strip_character)



df_test['caption'] = df_test['caption'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

df_test['caption'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in df_test['caption']]



counts_test = Counter()

for i, cap in enumerate(df_test['caption']):

    counts_test.update(cap.split())
vect_test = TfidfVectorizer(ngram_range = (1,4)).fit(df_test.caption)
# using the training data's words length for the testing as well to avoid the 

tokenizer_test = Tokenizer(num_words=len_token)

tokenizer_test.fit_on_texts(list(vect_test.get_feature_names()))

one_hot_res_test = tokenizer_test.texts_to_matrix(list(df_test.caption.values),mode='binary')
X_testpredict = one_hot_res_test;
np.ndim(one_hot_res_test)

one_hot_res_test.shape

testdata = test_ground_truth.copy()
testpredict_SVR_short = modelSVR_short.predict(X_testpredict)
testpredict_SVR_short.shape

type(testpredict_SVR_short)

testdata['short-term_memorability'] = testpredict_SVR_short
testpredict_RFR_long = rf3.predict(X_testpredict)
testdata['long-term_memorability'] = testpredict_RFR_long[:,1]
testdata.tail()
testdata.to_csv('finalgroundTruth_test.csv', encoding='utf-8', index=False)