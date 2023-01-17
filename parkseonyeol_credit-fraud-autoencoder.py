import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Dense, Input, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import regularizers
traindata = pd.read_csv('../input/creditcardfraud/creditcard.csv')
traindata.head(10)

#traindata= traindata.sample(frac = 0.1,random_state=1)
#샘플로 하이퍼 파라미터 찾기

traindata.describe()
normal_traindata = traindata[traindata['Class']==0]
fraud_traindata = traindata[traindata['Class']==1]
from sklearn.model_selection import train_test_split

normal_train, normal_test = train_test_split(normal_traindata, test_size=0.3, random_state=42)
testdata = pd.concat([fraud_traindata, normal_test])
X_normal_train = normal_train.drop(['Class'], axis=1)
Y_normal_train = normal_train['Class'] 
X_test = testdata.drop(['Class'], axis=1)
Y_test = testdata['Class'] 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_normal_train = sc.fit_transform(X_normal_train)
X_test = sc.transform(X_test)
'''from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_normal_train = sc.fit_transform(X_normal_train)
X_normal_test = sc.transform(X_normal_test)
X_fraud_test = sc.fit_transform(X_fraud_test)'''
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import regularizers
import tensorflow as tf

input_dim = X_normal_train.shape[1]
encoding_dim = int(input_dim * 2)
hidden_dim = int(encoding_dim * 2)
learning_rate = 1e-7
input_layer = Input(shape=(input_dim, ))

encoding_layer1 = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l2(learning_rate))(input_layer)
encoding_layer2 = Dense(hidden_dim, activation="relu")(encoding_layer1)

latent_view   = Dense(hidden_dim*2, activation='sigmoid')(encoding_layer2)

decoding_layer1 = Dense(hidden_dim, activation='tanh')(latent_view)
decoding_layer2 = Dense(encoding_dim, activation='relu')(decoding_layer1)

output_layer = Dense(input_dim)(decoding_layer2)

model = Model(input_layer, output_layer)
model.summary()
nb_epoch = 30
batch_size = 128

model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')

cp = ModelCheckpoint(filepath="autoencoder_fraud.h5", save_best_only=True, verbose=0)
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

history = model.fit(X_normal_train, X_normal_train, epochs=nb_epoch, batch_size=batch_size, shuffle=False, 
                    validation_data=(X_test, X_test), verbose=1, callbacks=[cp, tb]).history
model = load_model('autoencoder_fraud.h5')
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['X_normal_train', 'X_test'], loc='upper right');
pred = model.predict(X_test)
mse = np.mean(np.power(X_test - pred, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': Y_test})
error_df.describe()
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

threshold_fixed = 0.1

pred_y = [1 if e > threshold_fixed else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, pred_y)
LABELS = ["Normal","Fraud"]

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
tpos = conf_matrix[0][0]
fpos = conf_matrix[0][1]
tneg = conf_matrix[1][0]
fneg = conf_matrix[1][1]
print('정상 맞춘 확률', (tpos / (fpos + tpos)))
print('비정상 맞춘 확률', (fneg / (tneg + fneg)))
print('전체 정확도', ((tpos / (fpos + tpos)) + (fneg / (tneg + fneg))) / 2)
