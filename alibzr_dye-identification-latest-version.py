import numpy as np

import glob

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

from keras.callbacks import Callback

from keras.regularizers import l2

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras import regularizers

from itertools import tee
path = r'/kaggle/input/alidyes2/'



filenames = sorted(glob.glob(path + "/*.txt"))



labels = [{'AB92'},{'AB92','AG25'},{'AB92','DR23'},{'AB92','IN'},{'AB92','MO'},{'AB92','RHB'},{'AB92','RY17'},{'AG25'},{'AG25','MO'},{'AG25','RY17'},{'DR23'},{'DR23','AG25'},{'DR23','IN'},{'DR23,MO'},{'DR23','RHB'},{'DR23','RY17'},{'IN'},{'IN','AG25'},{'IN','MO'},{'IN','RY17'},{'MO'},{'RHB'},{'RHB','AG25'},{'RHB','IN'},{'RHB','MO'},{'RHB','RY17'},{'RY17'},{'RY17','MO'}]



dataframes = []



for filename in filenames:

    dataframes.append(pd.read_csv(filename, header=None)) 

    

for x in range(len(filenames)):

    dataframes[x].iloc[80,:] = [labels[x] for i in range(500)]

    

dataset = pd.concat(dataframes, axis=1, ignore_index=True)
path = r'/kaggle/input/dyeversionmix1/'



all_files = sorted(glob.glob(path + "/*.RLS"))



li = []



for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=None, delimiter='\t')

    li.append(df.iloc[0:396:5,1])



Test = pd.concat(li, axis=1, ignore_index=True)
mlb = MultiLabelBinarizer()

Y = mlb.fit_transform(dataset.iloc[80,:])

X = np.transpose(dataset.iloc[0:80,:])

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.1, random_state = 5)
class Metrics(Callback):

    

    def on_train_begin(self, logs={}):

         self.val_f1s = []

         self.val_recalls = []

         self.val_precisions = []

         self.val_accuracy = []



    def on_epoch_end(self, epoch, logs={}):

         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()

         val_targ = self.validation_data[1]

         _val_f1 = f1_score(val_targ, val_predict, average='weighted')

         _val_recall = recall_score(val_targ, val_predict, average='weighted')

         _val_precision = precision_score(val_targ, val_predict, average='weighted')

         _val_accuracy = accuracy_score(val_targ, val_predict)

         self.val_f1s.append(_val_f1)

         self.val_recalls.append(_val_recall)

         self.val_precisions.append(_val_precision)

         self.val_accuracy.append(_val_accuracy)

         print("f1: %f | precision: %f | recall %f | subset %f" %(_val_f1, _val_precision, _val_recall, _val_accuracy))

         return

 

metrics = Metrics()
model = Sequential()

model.add(Dense(32, input_dim=80, activation='relu',kernel_regularizer=l2(0.005)))

model.add(Dense(16, activation='relu',kernel_regularizer=l2(0.005)))

model.add(Dense(8, activation='sigmoid',))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit( xTrain, yTrain, validation_data=(xTest, yTest), epochs=500, batch_size=64, verbose=0, callbacks=[metrics])
preds = model.predict(Test.transpose())

preds[preds>=0.5] = 1

preds[preds<0.5] = 0

mlb.inverse_transform(preds)