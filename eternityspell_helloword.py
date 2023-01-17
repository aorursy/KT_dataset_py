import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense, Dropout, Lambda, Flatten

from keras.optimizers import Adam, RMSprop

from sklearn.model_selection import train_test_split



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"));



train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()



# test = pd.read_csv("../input/test.csv")

# print(test.shape)

# test.head()

X_train = pd.get_dummies(train.ix[:,1:].values.astype('float32')) # all pixel values

X_train = (train.ix[:,1:].values).astype('float32') # all pixel values

#y_train = train.ix[:,0].values.astype('int32') # only labels i.e targets digits

#X_test = pd.get_dummies(test.values.astype('float32'))



# X_train




