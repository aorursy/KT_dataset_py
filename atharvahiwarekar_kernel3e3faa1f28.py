import numpy as np 
import pandas as pd 
import os
import pandas
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding,Bidirectional,Dense,Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sklearn
from sklearn.model_selection import train_test_split
from keras.layers import LSTM
from keras.layers import Dropout

reader = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv',encoding='latin-1')
y = reader.iloc[:,0]
X = reader.iloc[:,-1]
y = np.array(y)
X = np.array(X)
y = y.reshape((1599999,1))
X = X.reshape((1599999,1))

X = X.tolist()
y = y.tolist()

temp = X
X = [' '.join(x) for x in temp]

for i in range(len(y)):
    y[i]=y[i][0]/2

tokenizer = Tokenizer(num_words=200,oov_token="<OOV>")
tokenizer.fit_on_texts(X)
# word_index = tokenizer.word_index
seq = tokenizer.texts_to_sequences(X)
pad = pad_sequences(seq,padding='post')

print(pad.shape)
x_train,x_test,y_train,y_test = train_test_split(pad,y,test_size=0.5)
print(x_test.shape)
!mkdir -p saved_model
print('done')
from keras.callbacks import ModelCheckpoint
def create_model():
    model = Sequential([
        Embedding(10000,1024,input_length=118),
        Bidirectional(LSTM(256,return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(256,return_sequences=True)),
        Dropout(0.2),
        Flatten(),
        Dense(units=3,activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

checkpoint_path = '/kaggle/working/saved_model/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=2
)

model = create_model()
print(model.summary())

model.save_weights(checkpoint_path.format(epoch=0))

history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=1024,verbose=1,callbacks=[cp_callback])
model.save('/kaggle/working/saved_model/model1.h5')
import matplotlib.pyplot as plt
from keras.models import load_model
model1 = load_model('/kaggle/working/saved_model/model1.h5')
model1.summary()
# print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
x_test[0]
temp=[]
for i in range(117):
    temp.append(x_test[0][i])
temp
temp=x_test[0]
temp = np.reshape(temp,(1,118))
x=model1.predict(temp)
print(x[0][0]," ",x[0][1]," ",x[0][2])
print(np.sum(x))
