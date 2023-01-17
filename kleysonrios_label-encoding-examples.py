# Classes to be encoded in numeric arrays
classes = ['yes','no','maybe','no','yes']
from sklearn.preprocessing import LabelEncoder

# LabelEncoder will generate an array where the class name is replaced by a given number that represent uniquely a given class.
le = LabelEncoder()
labels = le.fit_transform(classes)

print(classes)
print(labels)

from sklearn.preprocessing import LabelBinarizer

# LabelBinarizer will generate an One-hot enconding matrix.
# One hot encoding transforms categorical features to a format that works with some classification and regression algorithms.
# The algorithm will generate one boolean column for each category. Only one of these columns could take on the value 1 for each sample, representing the class.
# Hence, the term one hot encoding.

lb = LabelBinarizer()
labels = lb.fit_transform(classes)

print(classes)
print(labels)
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np

# One hot encoding using Keras

_classes = np.array(classes)
size = np.unique(_classes).size

le = LabelEncoder().fit(classes)
labels = np_utils.to_categorical(le.transform(classes), size)

print(classes)
print(labels)