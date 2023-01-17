import pandas as pd
# create the training & test sets
train = pd.read_csv("../input/train.csv")
test= pd.read_csv("../input/test.csv")
print(train.shape)
train.head()
print(test.shape)
test.head()
X_train = train.drop(['label'], axis=1).values.astype('float32') / 255 # all pixel values
y_train = train['label'].values.astype('int32') # only labels: targets digits
X_test = test.values.astype('float32') / 255
x_train = (train.iloc[:,1:].values).astype('float32') / 255 # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
x_test = test.values.astype('float32') / 255

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
import keras
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)