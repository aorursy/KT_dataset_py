import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs


print('data({0[0]},{0[1]})'.format(train.shape))
print (train.head())
import pylab
import numpy as np
x=train.values[300][1:]
np_x=np.reshape(x,(28,28))
pylab.gray()
pylab.imshow(np_x)
