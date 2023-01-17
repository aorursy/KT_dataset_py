import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input,Flatten,Dense,Conv2D,ReLU,BatchNormalization,MaxPool2D
from keras.optimizers import Adam
from keras.regularizers import l2
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)  
pd.set_option('max_rows', None)
data = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
Y = data['label'].to_numpy()
data = data.drop(columns=['label'])
X = data.to_numpy()
test = test.to_numpy()
X = X.reshape(-1,28,28,1)
test = test.reshape(-1,28,28,1)
X = X/255.0
test = test/255.0
indices = np.random.randint(low=0, high=42000, size=10)
for i in indices:
    plt.figure()
    plt.imshow(X[i,:,:,0], cmap='gray')
Y = to_categorical(Y, 10)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
i = Input(shape=(28,28,1))
m = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding='valid')(i)
m = BatchNormalization()(m)
m = ReLU()(m)
m = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(m)

m = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='valid')(m)
m = BatchNormalization()(m)
m = ReLU()(m)
m = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(m)

m = Flatten()(m)
m = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(m)
m = Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(m)
m = Dense(10, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(m)

model = Model(inputs=i, outputs=m)
model.summary()
opt = Adam(0.01)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
model.fit(X_train, Y_train, batch_size=64, epochs=100, verbose=2, validation_data=(X_test, Y_test))
preds = model.predict(test)
preds = preds.argmax(axis=1)
imageid = np.arange(1,28001)
output = pd.DataFrame({"ImageId": imageid,"Label": preds})
output.to_csv('submissions.csv', index=False)