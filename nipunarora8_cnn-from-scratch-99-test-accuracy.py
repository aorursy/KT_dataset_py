import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.utils import to_categorical
train=pd.read_csv('../input/digit-recognizer/train.csv')
y=train['label']
X=train.drop('label',axis=1).values/255.0

X=X.reshape(-1,28,28,1)

# y=to_categorical(y)
i=96

plt.imshow(X[i].reshape(28,28))

plt.title(y[i].argmax())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from keras.models import *

from keras.layers import *

from keras.callbacks import ModelCheckpoint,EarlyStopping
model=Sequential()



model.add(Conv2D(128, (3, 3), input_shape = (28,28,1)))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Conv2D(512, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))







model.add(Flatten())





model.add(Dense(512))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.5))





model.add(Dense(10, activation='softmax'))



model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.summary()
early_stopping_monitor = EarlyStopping(

    monitor='val_loss',

    patience=10,

    verbose=0,

    mode='auto',

    restore_best_weights=True

)

# callback=ModelCheckpoint('best_model.h5',monitor='val_loss',save_best_only=True,verbose=1,mode='auto')
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping_monitor])
import matplotlib.pyplot as plt

res=history.history



plt.plot(res['accuracy'],label="accuracy")

plt.plot(res['val_accuracy'],label="val acc")

plt.plot(res['loss'],label='loss')

plt.plot(res['val_loss'],label='val loss')

plt.legend()

plt.show()
test=pd.read_csv('../input/digit-recognizer/test.csv').values/255.0

test=test.reshape(-1,28,28,1)
model.evaluate(X_test,y_test)
pred=model.predict(test)
new_pred=[]

for i in pred:

    new_pred.append(i.argmax())
submit=pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submit['Label']=new_pred
submit.to_csv('submissions1.csv',index=False)