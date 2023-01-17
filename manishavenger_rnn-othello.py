import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,TimeDistributed
from keras.utils import np_utils

data=open("../input/othello/Othello.txt").read().lower()
char=sorted(list(set(data)))
charid={char:Id for Id,char in enumerate(char)}
#samplespertrain=100
id_to_char={Id:char for Id, char in enumerate(char)}
total_length=len(data)
unique_char=len(char)
charX=[]
y=[]
pause=total_length-100
for i in range(0,pause,1):
    inputChar=data[i:i+100]
    outputChar=data[i+100]
    charX.append([charid[char] for char in inputChar])
    y.append(charid[outputChar])

X = np.reshape(charX,(len(charX),100, 1)) #samples-timesteps-features
X = X/(len(char))
print(X.shape)
y = np_utils.to_categorical(y)
print(y.shape)
model = Sequential()
model.add(LSTM(256, input_shape=(100,1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,y,epochs=100,batch_size=50,validation_split=0.2)
import sys
start = np.random.randint(0, len(charX)-1)
pattern = charX[start]
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(char))
    prediction=model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = id_to_char[index]
    seq_in = [id_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    
