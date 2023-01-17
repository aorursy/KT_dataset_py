import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn import linear_model
from google.colab import files



uploaded = files.upload()
data = pd.read_csv('water_dataX.csv',header=0,encoding = 'unicode_escape')
data['Temp']=pd.to_numeric(data['Temp'],errors='coerce')

data['D.O. (mg/l)']=pd.to_numeric(data['D.O. (mg/l)'],errors='coerce')

data['PH']=pd.to_numeric(data['PH'],errors='coerce')

data['B.O.D. (mg/l)']=pd.to_numeric(data['B.O.D. (mg/l)'],errors='coerce')

data['CONDUCTIVITY (µmhos/cm)']=pd.to_numeric(data['CONDUCTIVITY (µmhos/cm)'],errors='coerce')

data['NITRATENAN N+ NITRITENANN (mg/l)']=pd.to_numeric(data['NITRATENAN N+ NITRITENANN (mg/l)'],errors='coerce')

data['TOTAL COLIFORM (MPN/100ml)Mean']=pd.to_numeric(data['TOTAL COLIFORM (MPN/100ml)Mean'],errors='coerce')

data.dtypes
start=2

end=1779
station=data.iloc [start:end ,0]

location=data.iloc [start:end ,1]

state=data.iloc [start:end ,2]

do= data.iloc [start:end ,4].astype(np.float64)

value=0

ph = data.iloc[ start:end,5]  

co = data.iloc [start:end ,6].astype(np.float64)   

  

year=data.iloc[start:end,11]

tc=data.iloc [2:end ,10].astype(np.float64)





bod = data.iloc [start:end ,7].astype(np.float64)

na= data.iloc [start:end ,8].astype(np.float64)
na.dtype



data.head()
data=pd.concat([station,location,state,do,ph,co,bod,na,tc,year],axis=1)

data. columns = ['station','location','state','do','ph','co','bod','na','tc','year']

data['npH']=data.ph.apply(lambda x: (100 if (8.5>=x>=7)  

                                 else(80 if  (8.6>=x>=8.5) or (6.9>=x>=6.8) 

                                      else(60 if (8.8>=x>=8.6) or (6.8>=x>=6.7) 

                                          else(40 if (9>=x>=8.8) or (6.7>=x>=6.5)

                                              else 0)))))



data['ndo']=data.do.apply(lambda x:(100 if (x>=6)  

                                 else(80 if  (6>=x>=5.1) 

                                      else(60 if (5>=x>=4.1)

                                          else(40 if (4>=x>=3) 

                                              else 0)))))



data['nco']=data.tc.apply(lambda x:(100 if (5>=x>=0)  

                                 else(80 if  (50>=x>=5) 

                                      else(60 if (500>=x>=50)

                                          else(40 if (10000>=x>=500) 

                                              else 0)))))



data['nbdo']=data.bod.apply(lambda x:(100 if (3>=x>=0)  

                                 else(80 if  (6>=x>=3) 

                                      else(60 if (80>=x>=6)

                                          else(40 if (125>=x>=80) 

                                              else 0)))))



data['nec']=data.co.apply(lambda x:(100 if (75>=x>=0)  

                                 else(80 if  (150>=x>=75) 

                                      else(60 if (225>=x>=150)

                                          else(40 if (300>=x>=225) 

                                              else 0)))))



data['nna']=data.na.apply(lambda x:(100 if (20>=x>=0)  

                                 else(80 if  (50>=x>=20) 

                                      else(60 if (100>=x>=50)

                                          else(40 if (200>=x>=100) 

                                              else 0)))))
data.head()

data.dtypes
data['wph']=data.npH * 0.165

data['wdo']=data.ndo * 0.281

data['wbdo']=data.nbdo * 0.234

data['wec']=data.nec* 0.009

data['wna']=data.nna * 0.028

data['wco']=data.nco * 0.281

data['wqi']=data.wph+data.wdo+data.wbdo+data.wec+data.wna+data.wco 
data
#ag=data.groupby('station')['wqi'].mean()
#data=ag.reset_index(level=0,inplace=False)
data
station=data['station'].values

AQI=data['wqi'].values

data['wqi']=pd.to_numeric(data['wqi'],errors='coerce')

data['station']=pd.to_numeric(data['station'],errors='coerce')
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(year,AQI, color='red')

plt.show()
data
data = data[np.isfinite(data['wqi'])]

data.head()

data = data.replace('NAN','NaN')

data = data.dropna()
cols =['station']

y = data['wqi']

x=data[cols]

plt.scatter(x,y)

plt.show()
import matplotlib.pyplot as plt

data=data.set_index('station')

data.plot(figsize=(15,6))

plt.show()
from sklearn import neighbors,datasets

data=data.reset_index(level=0,inplace=False)

data





cols =['station']



y = data['wqi']

x=data[cols]

print(x.isna().sum())
reg=linear_model.LinearRegression()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)



reg.fit(x_train,y_train)



a=reg.predict(x_test)
a
y_test
from sklearn.metrics import mean_squared_error

print('mse:%.2f'%mean_squared_error(y_test,a))
#print("R^2 score for liner regression: ", abs(reg.score(x_test, y_test)))
#encoding the states
ag1 = data
ag1.head()
ag2=ag1.drop(['station','location','do','ph','bod','na','nbdo','nec','nna','wph','wdo','wbdo','wec','wna','wco'],axis = 1)
ag2.head()
ag2 = ag2.drop(['co','tc','npH','ndo','nco'],axis = 1)
ag2.head()
ag2
data=ag2.reset_index(level=0,inplace=False)
data
data = data.replace(',', np.nan).dropna()

data = data.replace('/', np.nan).dropna()

data = data.replace('(', np.nan).dropna()

data = data.replace(')', np.nan).dropna()
data.head()
data
data
list = ["DAMAN & DIU","GOA","MAHARASHTRA","KERALA","ANDHRA PRADESH","KARNATAKA","ODISHA","PONDICHERRY","TAMILNADU","TAMIL NADU","PUNJAB","HARYANA","RAJASTHAN","HIMACHAL PRADESH","MEGHALAYA","MIZORAM","TRIPURA","ORISSA","GUJARAT","MANIPUR"," H.P.","MADHYA PRADESH",""]
def common_member(a, b): 

    a_set = set(a) 

    b_set = set(b) 

    if (a_set & b_set): 

        data['state'][i] = a_set & b_set

    else: 

        data['state'][i] = np.nan

for i in range(1777):

    li=[]

    li = data['state'][i].split(', ')

    common_member(li,list)
data
data = data.dropna(axis = 0)
data.head(50)
data.tail(50)
data1 = data 
data.dtypes
data1.to_csv('data1.csv')
d = pd.read_csv('data1.csv',header=0,encoding = 'unicode_escape')
d
d.dtypes
d = d.drop(['Unnamed: 0','index'],axis = 1)
d
for i in range(1115):

    x = d['state'][i]

    d['state'][i] = str(x)
d.to_csv('state_wqi_year.csv')
d = pd.read_csv('state_wqi_year.csv',header=0, encoding = 'unicode_escape')
d.dtypes
d.head()
d = d.drop(['Unnamed: 0'],axis = 1)
labels = d['state']
labels
features = d
features
features = features.drop(['state'],axis = 1)
#f1 = features.values.reshape(-1,1)
d.head()
features.head(50)
from keras.utils import to_categorical

le = LabelEncoder()

#features_e = le.fit_transform(features)

#features_o = to_categorical(features_e)

#labels_e = le.fit_transform(labels)

#labels_o = to_categorical(labels_e)
labels.shape
labels.head()
type(features.shape)
features.head()
features.drop('state')
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,random_state = 0) 
features.shape
#print("X_trian shape --> {}".format(x_train.shape))

#print("y_train shape --> {}".format(y_train.shape))

#print("X_test shape --> {}".format(x_test.shape))

#print("y_test shape --> {}".format(y_test.shape))



y_train.to_csv('y_train.csv')



y_test.to_csv('y_test.csv')
labels = pd.get_dummies(labels,drop_first=True)
labels
from keras.utils import to_categorical

le = LabelEncoder()

ytrain_e = le.fit_transform(y_train)

ytrain_o = to_categorical(ytrain_e)

ytest_e = le.fit_transform(y_test)

ytest_o = to_categorical(ytest_e)
y_train[:5],y_test[:5]
x_train.shape
x_test.shape
y_train.shape
y_test.shape
print(ytrain_o[5])

print(ytest_o[4])

print(a[4])
reg=linear_model.LinearRegression()



reg.fit(x_train,y_train)



a=reg.predict(x_test)

a
print(a[0])
a.shape
ytrain_o.shape
ytest_o.shape
from sklearn.metrics import mean_squared_error

print('mse:%.2f'%mean_squared_error(y_test,a))
print("R^2 score for liner regression: ", abs(reg.score(x_train, y_train)))
!sudo apt-get install build-essential swig

!curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

!pip install auto-sklearn

import autosklearn.classification
!pip install auto-sklearn

import autosklearn.classification as classifier

# ac = classifier.AutoSklearnClassifier()

ac = classifier.AutoSklearnClassifier(time_left_for_this_task=520,per_run_time_limit=40)
ac.fit(x_train,y_train)
ac_pred = ac.predict(x_test)
ac_pred.shape
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

metrics.accuracy_score(y_test, ac_pred)

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.text import text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences



from keras.models import Model

from keras.models import Sequential



from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D

from keras.layers import Reshape, Flatten, Dropout, Concatenate

from keras.layers import SpatialDropout1D, concatenate

from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D



from keras.callbacks import Callback

from keras.optimizers import Adam



from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.models import load_model

from keras.utils.vis_utils import plot_model
MAX_NB_WORDS = 80000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
train_sequences = tokenizer.texts_to_sequences(x_train)

test_sequences = tokenizer.texts_to_sequences(x_test)
MAX_LENGTH = 500

padded_train_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH)

padded_test_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH)
def get_simple_rnn_model():

    embedding_dim = 800

    embedding_matrix = np.random.random((MAX_NB_WORDS, embedding_dim))

    

    inp = Input(shape=(MAX_LENGTH, ))

    x = Embedding(input_dim=MAX_NB_WORDS, output_dim=embedding_dim, input_length=MAX_LENGTH, 

                  weights=[embedding_matrix], trainable=True)(inp)

    x = SpatialDropout1D(0.3)(x)

    x = Bidirectional(GRU(100, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)

    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    outp = Dense(20, activation="sigmoid")(conc)

    

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='categorical_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])

    return model



rnn_simple_model = get_simple_rnn_model()
x_train.shape
y_train.shape
filepath="./weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')



batch_size = 16

epochs = 1



history = rnn_simple_model.fit(x=x_train, 

                    y=y_train, 

                    validation_data=(x_test, y_test), 

                    batch_size=batch_size, 

                    callbacks=[checkpoint], 

                    epochs=epochs)
best_rnn_simple_model = load_model('./weights-improvement-01-0.8262.hdf5')



y_pred_rnn_simple = best_rnn_simple_model.predict(padded_test_sequences, batch_size=16)



y_pred_rnn_simple = pd.DataFrame(y_pred_rnn_simple, columns=['prediction'])

y_pred_rnn_simple['prediction'] = y_pred_rnn_simple['prediction'].map(lambda p: 1 if p >= 0.5 else 0)

y_pred_rnn_simple.to_csv('./y_pred_rnn_simple.csv', index=False)





y_pred_rnn_simple = pd.read_csv('./y_pred_rnn_simple.csv')

print(accuracy_score(y_test_oh, y_pred_rnn_simple))
ag=data.groupby('year')['wqi'].mean()



ag.head()



data=ag.reset_index(level=0,inplace=False)

data



year=data['year'].values

AQI=data['wqi'].values

data['wqi']=pd.to_numeric(data['wqi'],errors='coerce')

data['year']=pd.to_numeric(data['year'],errors='coerce')
data = data[np.isfinite(data['wqi'])]

data.head()



cols =['year']

y = data['wqi']

x=data[cols]
plt.scatter(x,y)

plt.show()



import matplotlib.pyplot as plt

data=data.set_index('year')

data.plot(figsize=(15,6))

plt.show()
from sklearn import neighbors,datasets

data=data.reset_index(level=0,inplace=False)

data

cols =['year']



y = data['wqi']

x=data[cols]
reg=linear_model.LinearRegression()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)



reg.fit(x_train,y_train)



a=reg.predict(x_test)

a

y_test
from sklearn.metrics import mean_squared_error

print('mse:%.2f'%mean_squared_error(y_test,a))
dt = pd.DataFrame({'Actual': y_test, 'Predicted': a}) 



x = (x - x.mean()) / x.std()

x = np.c_[np.ones(x.shape[0]), x]

x
print("R^2 score for liner regression: ", abs(reg.score(x_test, y_test)))