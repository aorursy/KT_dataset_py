# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from collections import Counter
fg_unet = pd.read_csv("/kaggle/input/mnist-w-fgunet-output/submission.csv")

vgg = pd.read_csv("/kaggle/input/mnist-w-vgg16-output/submission.csv")

resnet = pd.read_csv("/kaggle/input/mnist-w-resnet-output/submission.csv")
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

sub = pd.read_csv("/kaggle/input/mnist-w-resnet-output/submission.csv")
# Ensemble 1: majority vote

majority = sub.copy()



no_maj = []

unequal = []

for i in range(len(vgg)):

    lst = [fg_unet.iloc[i].Label, vgg.iloc[i].Label, resnet.iloc[i].Label]

    if not all(ele == lst[0] for ele in lst):

        unequal.append(i)

        count = Counter(lst).most_common()

        if len(count)==len(lst):

            no_maj.append(i)

            majority.iloc[i].Label = lst[-1] # resnet had the highest standalone accuracy

            img = test.iloc[i,:].values

            img = img.reshape(28, 28)

            print(i, lst)

            plt.imshow(img)

            plt.show()

        else:

            majority.iloc[i].Label = count[0][0]

    else:

        majority.iloc[i].Label = lst[0]

print("Number of rows not all equal: ", len(unequal))

print("Number of rows with no majority: ", len(no_maj))



majority.to_csv("submission_maj.csv", index=False)
from keras.models import load_model

from keras.utils import to_categorical
fg_unet = load_model("/kaggle/input/mnist-w-fgunet-output/best_model.h5")

vgg = load_model("/kaggle/input/mnist-w-vgg16-output/best_model.h5")

resnet = load_model("/kaggle/input/mnist-w-resnet-output/best_model.h5")
X = [train.iloc[i,1:].values for i in range(len(train))]

X = [x.reshape(28,28) for x in X]

X_28 = [x.reshape(28,28,1,1) for x in X]

X_28 = np.array(X_28)

X = [np.pad(x, 2) for x in X]

X = np.array(X)

X = X.reshape(X.shape[0],X.shape[1], X.shape[2],1)

X = np.repeat(X, 3, axis=-1)
X_test = [test.iloc[i,:].values for i in range(len(test))]

X_test = [x.reshape(28,28) for x in X_test]

X_test_28 = [x.reshape(28,28,1,1) for x in X_test]

X_test_28 = np.array(X_test_28)

X_test = [np.pad(x, 2) for x in X_test]

X_test = np.array(X_test)

X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], X_test.shape[2],1)

X_test = np.repeat(X_test, 3, axis=-1)
X.shape, X_28.shape, X_test.shape, X_test_28.shape
f_y_train = fg_unet.predict(X_28, verbose=1)

v_y_train = vgg.predict(X, verbose=1)

r_y_train = resnet.predict(X, verbose=1)
f_y_test = fg_unet.predict(X_test_28, verbose=1)

v_y_test = vgg.predict(X_test, verbose=1)

r_y_test = resnet.predict(X_test, verbose=1)
n_classes = 10

y = [train.iloc[i,0] for i in range(len(train))]

y = np.array(y)

print(np.unique(y, return_counts=True))

y = to_categorical(y, num_classes=n_classes)

y.shape
# Ensemble 2: Ridge Regression



from sklearn.linear_model import Ridge

X = np.hstack([f_y_train, v_y_train, r_y_train])

rid = Ridge()

rid.fit(X, y)
X_pred = np.hstack([f_y_test, v_y_test, r_y_test])
y_pred = rid.predict(X_pred)

y_pred = np.argmax(y_pred, axis=-1)

print(y_pred.shape)
ridge = sub.copy()

for i in range(len(y_pred)):

    ridge.iloc[i].Label = y_pred[i]

ridge.to_csv("submission_ridge.csv", index=False)
# Ensemble 3: Generic Weighted Average

from keras import Input, Model

from keras.layers import Conv1D, Softmax, Flatten

from keras.initializers import Constant



init = Constant(.333)



X = Input(shape=(10,3))

out = Conv1D(1, 1, use_bias=False, kernel_initializer=init, kernel_regularizer='l2')(X)

out = Flatten()(out)

out = Softmax()(out)

model = Model(X, out)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.summary()
X = np.stack([f_y_train, v_y_train, r_y_train], axis=-1)

X_pred = np.stack([f_y_test, v_y_test, r_y_test], axis=-1)

X.shape, X_pred.shape
model.fit(X, y, epochs=10)
y_pred = model.predict(X_pred)

print(y_pred[0])

y_pred = np.argmax(y_pred, axis=-1)

print(y_pred.shape)
weighted = sub.copy()

for i in range(len(y_pred)):

    weighted.iloc[i].Label = y_pred[i]

weighted.to_csv("submission_weighted.csv", index=False)
weights = model.layers[1].get_weights()

weights = weights/np.sum(weights)

weights
# Ensemble 4 & 5: single transferable vote & runoff

!pip install pyrankvote
import pyrankvote

from pyrankvote import Candidate, Ballot
candidates = [Candidate(i) for i in range(10)]
single = sub.copy()

runoff = sub.copy()

for i in range(len(X_pred)):

    ballots = []

    for j in range(3):

        ballot = np.argsort(X_pred[i,:,j])

        ballot = np.flip(ballot)

        

        ballots.append(Ballot(ranked_candidates=[candidates[i] for i in ballot]))

    

    election_result = pyrankvote.single_transferable_vote(candidates, ballots, number_of_seats=1)

    winners = election_result.get_winners()

    

    single.iloc[i].Label = winners[0].name

    

    election_result = pyrankvote.instant_runoff_voting(candidates, ballots)

    winners = election_result.get_winners()

    

    runoff.iloc[i].Label = winners[0].name



single.to_csv("submission_single.csv", index=False)

runoff.to_csv("submission_runoff.csv", index=False)