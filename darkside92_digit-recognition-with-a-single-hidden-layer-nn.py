import numpy as np

import pandas as pd

%matplotlib inline

         

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from matplotlib.pyplot import imshow



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
np.random.seed(1337)


train = pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

train.head(10)
test.head(10)
train.groupby(['label'],axis=0).size()
X_train = train.drop(['label'],axis=1).astype('float32').values # all pixel values

y_train = train['label'].astype('int32').values

X_test = test.values.astype('float32')
X_train = X_train / 255

X_train  = X_train.round()

X_test = X_test / 255

X_test  = X_test.round()



y_train = np_utils.to_categorical(y_train)
pixelcount = X_train.shape[1]
model = Sequential()

model.add(Dense(pixelcount, input_dim=pixelcount, kernel_initializer='normal', activation='relu'))

model.add(Dense(10, kernel_initializer='normal', activation='softmax'))

#Compile

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train,epochs=10, batch_size=200, verbose=2)
preds = model.predict(X_test)
test = np.argmax(preds,axis=1)
pd.DataFrame({"ImageId": list(range(1,len(test)+1)), "Label": test}).to_csv("mysubmit.csv", index=False, header=True)