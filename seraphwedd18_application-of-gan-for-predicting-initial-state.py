# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/conways-reverse-game-of-life-2020/train.csv")

test = pd.read_csv("../input/conways-reverse-game-of-life-2020/test.csv")

ss = pd.read_csv("../input/conways-reverse-game-of-life-2020/sample_submission.csv")
class GameOfLife():

    

    def __init__(self, start_array):

        '''Start array is a 25x25 array of ones and zeros.'''

        self.array = start_array

        self.forward_pass()

        self.padded_array = self.pad(self.array)

    

    def pad(self, array):

        temp = np.zeros((27, 27))

        for y in range(27):

            for x in range(27):

                temp[y, x] = array[(y-1)%25, (x-1)%25]

        return temp

    

    def forward_pass(self):

        try:

            self.Y = self.X

        except:

            self.Y = self.array

        

        self.X = np.zeros((25, 25))

        

        for x in range(25):

            for y in range(25):

                self.X[y][x] = self.get_surround(self.Y, x, y)

        return self.X, self.Y

    

    def get_surround(self, array, x, y):

        count = 0 

        for cx in range(x-1, x+2):

            for cy in range(y-1, y+2):

                if x == cx and y == cy:

                    pass

                else:

                    count += array[cy%25][cx%25]

        if count == 2:

            return array[y][x]

        elif count == 3:

            return 1

        else:

            return 0

    

    def printxy(self):

        print('X Values:')

        print(*self.X.astype('uint8').tolist(), sep='\n')

        print('Y Values:')

        print(*self.Y.astype('uint8').tolist(), sep='\n')

        
print("Test for binary values:")

cols = [col for col in train.columns if col.startswith('start')]

x = np.array(train[cols].values).reshape(-1, 25, 25)[3]

h = GameOfLife(x)

h.printxy()
print("Test visualization and forward_pass:")

plt.subplot(231)



for i in range(3):

    _, _ = h.forward_pass()

    plt.subplot(2, 3, i+1, label='X Values:')

    plt.imshow(h.X)

    plt.subplot(2, 3, i+4, label='Y Values:')

    plt.imshow(h.Y)

plt.show()

plt.close()
def train_batch_generator(size, df):

    batch_size = size//5

    cols = [x for x in df.columns if x.startswith('start')]

    delta = df.delta

    ids = df.id

    zeroX = np.zeros((27, 27, 1))

    zeroY = np.zeros(625)

    all_values = df[cols].values.reshape(-1, 25, 25)

    

    for i in range(0, len(ids), batch_size):

        X = [zeroX for _ in range(5*batch_size)]

        Y = [zeroY for _ in range(5*batch_size)]

        

        for j in range(batch_size):

            try:

                item = GameOfLife(all_values[i+j])

                for k in range(5):

                    X[5*j+k] = item.pad(item.X).reshape((27, 27, 1))

                    Y[5*j+k] = item.Y.reshape(625)

                    item.forward_pass()

            except Exception as e:

                print('computing error...', e)

                yield np.array(X), np.array(Y)

                return

        yield np.array(X), np.array(Y)
print("Generator test:")

import time

start, end = time.time(), time.time()

for x, y in train_batch_generator(50, train[:120]):

    start, end = end, time.time()

    print(x.shape, end - start)
def test_batch_generator(batch_size, df):

    cols = [x for x in df.columns if x.startswith('stop')]

    delta = df.delta

    ids = df.id

    zeroX = np.zeros((27, 27, 1))

    all_values = df[cols].values.reshape(-1, 25, 25)

    

    for i in range(0, len(ids), batch_size):

        X = []

        

        for j in range(batch_size):

            try:

                item = GameOfLife(all_values[i+j])

                X.append(item.pad(item.X).reshape((27, 27, 1)))

                

            except Exception as e:

                print('computing error...', e)

                yield np.array(X)

                return

        yield np.array(X)
print("Generator test:")

import time

start, end = time.time(), time.time()

for x in test_batch_generator(50, train[:120]):

    start, end = end, time.time()

    print(x.shape, end - start)
import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, multiply, add, MaxPooling2D, Reshape

from tensorflow.keras.losses import BinaryCrossentropy as BC

from tensorflow.keras.losses import CategoricalCrossentropy as CC

from tensorflow.keras.optimizers import Adam



np.random.seed(171)



def custom_loss_func(y_true, y_pred):

    error = (y_true - y_pred)

    return error**2



def predictor(lr=0.01):

    

    i = Input((27, 27, 1))

    x1 = Conv2D(64, (3, 3), activation='relu', padding='valid')(i)

    x2 = Conv2D(64, (3, 3), activation='relu', padding='valid')(i)

    x3 = Conv2D(64, (3, 3), activation='sigmoid', padding='valid')(i)

    x = multiply([x3, add([x1, x2])])

    

    x = MaxPooling2D((3, 3))(x)

    x = Reshape((64, 64, 1))(x)

    

    x = Conv2D(128, (9, 9), activation='relu', padding='valid')(x)

    x1 = Conv2D(32, (9, 9), activation='relu', padding='valid')(x)

    x2 = Conv2D(32, (9, 9), activation='sigmoid', padding='valid')(x)

    x = multiply([x1, x2])

    x = MaxPooling2D((3, 3))(x)

    x = Flatten()(x)

    

    x = Dense(5000, activation='relu')(x)

    x = Dense(2500, activation='relu')(x)

    x = Dense(1250, activation='relu')(x)

    

    o = Dense(625, activation='sigmoid')(x)

    

    model = Model(inputs=i, outputs=o)

    opt = Adam(lr=lr)

    #loss = BC(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    loss = CC(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    model.compile(optimizer=opt,

                  loss=loss,

                  metrics=['accuracy', 'mse'])

    

    return model



model = predictor()

model.summary()

def roundoff(array):

    #Array should be of shape (625)

    arr = array.reshape((625))

    mean = np.quantile(arr, 0.5) if max(arr) < 0.3 else 0.5

    f = lambda x: x>mean

    v = np.vectorize(f)

    return v(arr)
def test_model(model, n=1):

    y = np.array([[np.random.randint(0, 2) for _ in range(25)] for _ in range(25)])

    test_acc = GameOfLife(y)

    

    for _ in range(5): #Preconditioning

        _, _ = test_acc.forward_pass()

        

    for _ in range(n):

        x, y = test_acc.forward_pass()

        pred_y = model.predict(test_acc.pad(x).reshape(-1, 27, 27, 1))

        plt.subplot(221)

        plt.subplot(2, 2, 1, label=f'X Values:')

        plt.imshow(x.reshape((25, 25)))

        plt.subplot(2, 2, 2, label=f'Y Values:')

        plt.imshow(y.reshape((25, 25)))

        plt.subplot(2, 2, 3, label=f'Predicted Y Values:')

        plt.imshow(pred_y.reshape((25, 25)))

        plt.subplot(2, 2, 4, label='Rounded Predicted Y Values:')

        plt.imshow(roundoff(pred_y).reshape((25, 25)))

        plt.show()

        print("Accuracy:", sum(sum(np.equal(roundoff(pred_y).reshape((25, 25)), y)))/625)

test_model(model, 10)
from sklearn.model_selection import train_test_split as tts

from tensorflow.keras.callbacks import ReduceLROnPlateau as rlrp

from tensorflow.keras.callbacks import EarlyStopping as es



call_rlrp = rlrp(monitor='val_accuracy', patience=50, min_lr=1e-8, factor=0.2, verbose=1)

call_es = es(monitor='val_mse', patience=100, mode='auto', verbose=1)



print("Begin training:\t%s\n" %time.ctime())



model = predictor(.0005)

start, end = time.time(), time.time()

for n, (x, y) in enumerate(train_batch_generator(250000, train)):

    print(f"Training new batch {n+1}...")

    tx, vx, ty, vy =tts(x, y, test_size=0.2, shuffle=True, random_state=172)

    model.fit(tx, ty,

              validation_data=(vx, vy),

              batch_size=256, epochs=300,

              callbacks=[call_rlrp, call_es],

              verbose=1)

    model.evaluate(vx, vy, verbose=1)

    start, end = end, time.time()

    print("#"*72)

    print("Total time for batch Training: %.2f" %(end - start))

    test_model(model, 3)

    print("#"*72)

model.save('train_model.h5')

print("\nEnd training:\t%s" %time.ctime())
preds = []

test_x = []

for x in test_batch_generator(10, train[:100]):

    temp = model.predict(x)

    try:

        preds = np.concatenate([preds, temp], axis=0)

        test_x = np.concatenate([test_x, x], axis=0)

    except Exception as e:

        print(e)

        preds = temp

        test_x = x

        

preds = np.array(preds)

print(preds.shape)

print("Test visualization of predictions:")



for i in range(0, 96, 3):

    print(f"Test data from i={i} to i={i+3}:")

    plt.subplot(331)

    plt.subplot(3, 3, 1, label=f'X {i} Values:')

    plt.imshow(test_x[i][1:26, 1:26].reshape((25, 25)))

    plt.subplot(3, 3, 4, label=f'Y {i} Values:')

    plt.imshow(preds[i].reshape((25, 25)))

    plt.subplot(3, 3, 7, label=f'Y {i} Values:')

    plt.imshow(roundoff(preds[i]).reshape((25, 25)))

    

    plt.subplot(3, 3, 2, label=f'X {i} Values:')

    plt.imshow(test_x[i+1][1:26, 1:26].reshape((25, 25)))

    plt.subplot(3, 3, 5, label=f'Y {i} Values:')

    plt.imshow(preds[i+1].reshape((25, 25)))

    plt.subplot(3, 3, 8, label=f'Y {i} Values:')

    plt.imshow(roundoff(preds[i+1]).reshape((25, 25)))

    

    plt.subplot(3, 3, 3, label=f'X {i} Values:')

    plt.imshow(test_x[i+2][1:26, 1:26].reshape((25, 25)))

    plt.subplot(3, 3, 6, label=f'Y {i} Values:')

    plt.imshow(preds[i+2].reshape((25, 25)))

    plt.subplot(3, 3, 9, label=f'Y {i} Values:')

    plt.imshow(roundoff(preds[i+2]).reshape((25, 25)))

    

    plt.show()

    plt.close()
preds = []

print("Compute for all delta >= 1...")

for x in test_batch_generator(10000, test):

    temp = model.predict(x, verbose=2)

    try:

        preds = np.concatenate([preds, temp], axis=0)

    except Exception as e:

        print(e)

        preds = temp





preds = np.array(preds)

print(preds.shape)

for i in range(2, 6):

    print(f"Computing for all delta >= {i}...")

    indices = test.index[test.delta >= i].tolist()

    preds[indices] = [roundoff(x) for x in model.predict(np.array(

        [h.pad(array.reshape((25, 25))) for array in preds[indices]])

                                                        )]
ids = test.id



submission = pd.DataFrame([roundoff(x) for x in preds], index=ids)

submission.columns = ['start_'+str(c) for c in range(625)]

submission = submission.astype('uint32')

submission.head()

submission.to_csv("submission.csv", index=True)
submission.head()
submission.describe()