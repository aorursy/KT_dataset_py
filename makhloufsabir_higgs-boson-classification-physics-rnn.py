import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder,normalize,MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
import seaborn as sns


import tensorflow as tf
# import tensorflow as tf

# # GPU device Check.
# device_name = tf.test.gpu_device_name()
# if device_name == '/device:GPU:0':
#     print('Found GPU at: {}'.format(device_name))
# else:
#     raise SystemError('GPU device not found')
    
# import torch

# # If there's a GPU available...
# if torch.cuda.is_available():    

#     # PyTorch use the GPU.    
#     device = torch.device("cuda")

#     print('There are %d GPU(s) available.' % torch.cuda.device_count())

#     print('We will use the GPU:', torch.cuda.get_device_name(0))

# # If not...
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")
# Reading data
train = pd.read_csv('../input/higgs-boson/training.zip')
test = pd.read_csv('../input/higgs-boson/test.zip')

print(train.shape,test.shape)
train
print(train.columns.values,'\n')
print(test.columns.values)
train = train.drop(['Weight'], axis=1)
print(train['Label'].value_counts())

rcParams['figure.figsize'] = 10,5
sb.barplot(x = train['Label'].value_counts().index, y = train['Label'].value_counts().values)
plt.title('Label counts')
plt.show()
# getting dummy variables column

enc = LabelEncoder()

train['Label'] = enc.fit_transform(train['Label'])
train.head()
y = train["Label"]
X = train
X_test = test
X.set_index(['EventId'],inplace = True)
X_test.set_index(['EventId'],inplace = True)
X = X.drop(['Label'], axis=1)

X.head()
X_test.head()
train.describe()
#Normalizing

from sklearn.preprocessing import normalize

X = normalize(X)
X_test = normalize(X_test)
# from sklearn.model_selection import StratifiedKFold
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D,Activation, BatchNormalization
# from tensorflow.keras.models import Sequential

# BATCH_SIZE = 8
# n_fold = 5

# kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
# cvscores = []  
# for train, test in kfold.split(X, y): 
#   # create model 
#     model = Sequential() 
#     model.add(Dense(1024, input_dim=30, activation='relu'))
#     model.add(Dropout(0.8)) 
#     model.add(Dense(1024, activation='relu')) 
#     model.add(Dropout(0.8)) 
#     model.add(Dense(512, activation='relu')) 
#     model.add(Dropout(0.8)) 
#     model.add(Dense(2,activation='softmax'))
#     # Compile model
#     opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01, amsgrad=False)
#     model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#     # Fit the model
#     model.fit(X[train], y[train],validation_data=(X[train], y[train]), epochs=10, batch_size=BATCH_SIZE, verbose=0)
#     # evaluate the model
#     scores = model.evaluate(X[test], y[test], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100) 
      
#     #prediction     
#     prediction = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)   
    
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))) 
#K Fold Cross Validation

from sklearn.model_selection import KFold


kf = KFold(n_splits=5, random_state=2020, shuffle=True)

for train_index, val_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", val_index)
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
# import xgboost as xgb

# dtrain = xgb.DMatrix(X_train, label=y_train)
# dvalid = xgb.DMatrix(X_val, label=y_val)
# watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# xgb_pars = {'min_child_weight': 100, 'eta': 0.04, 'colsample_bytree': 0.8, 'max_depth': 100,
#             'subsample': 0.75, 'lambda': 2, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1, 'gamma' : 0,
#             'eval_metric': 'rmse', 'objective': 'reg:linear'}    

# model = xgb.train(xgb_pars, dtrain, 500, watchlist, early_stopping_rounds=250,
#                   maximize=False, verbose_eval=15) 
# dtest = xgb.DMatrix(X_test)

# prediction = model.predict(dtest)  
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
#reshape for rnn

X_train = X_train.reshape(-1, 1, 30)
X_val  = X_val.reshape(-1, 1, 30)
y_train = y_train.values #convert pd to array
y_train = y_train.reshape(-1, 1,)
y_val = y_val.values #convert pd to array
y_val = y_val.reshape(-1, 1,)
X_train.shape
from tensorflow.keras.layers import Conv2D,LSTM,LeakyReLU, MaxPooling2D,Concatenate,Input, Dropout, Flatten, Dense, GlobalAveragePooling2D,Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model


  # create model
    

#input 
input_layer = Input(shape=(1,30))
main_rnn_layer = LSTM(64, return_sequences=True, recurrent_dropout=0.2)(input_layer)

    
#output
rnn = LSTM(32)(main_rnn_layer)
dense = Dense(128)(rnn)
dropout_c = Dropout(0.3)(dense)
classes = Dense(1, activation= LeakyReLU(alpha=0.1),name="class")(dropout_c)

model = Model(input_layer, classes)

# Compile model
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, factor=0.6),
             EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model.compile(loss=[tf.keras.losses.MeanSquaredLogarithmicError(),tf.keras.losses.MeanSquaredLogarithmicError()], optimizer="adam")


model.summary()
# Fit the model
history = model.fit(X_train, y_train, 
          epochs = 500, 
          batch_size = 16, 
          validation_data=(X_val,  y_val), 
          callbacks=callbacks)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()
X_test.shape
model.load_weights("best_model.h5")

test = X_test #convert pd to array
test = test.reshape(-1, 1,30)


predictions = model.predict(test)
print(predictions.shape)
print(predictions)
sub = pd.read_csv('../input/higgs-boson/random_submission.zip')
sub
type(predictions)
pred = np.where(predictions > 0.5, 1, 0)
pred
test_predict = pd.Series(pred[:,0])
test_predict
test_predict = pd.DataFrame({"EventId":sub['EventId'],"RankOrder":sub['RankOrder'],"Class":test_predict})
test_predict
test_predict = test_predict.replace(1,'s')
test_predict = test_predict.replace(0,'b')
test_predict
test_predict['RankOrder'] = test_predict['Class'].argsort().argsort() + 1 # +1 to start at 1
test_predict
test_predict.to_csv("submission.csv",index=False)