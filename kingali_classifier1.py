import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
#print (train.iloc[1:47999,1:784])
train_data=np.array(train)[:,1:785]
train_labels=np.array(train)[:,0]
train_labels=train_labels.reshape((-1,1))
test_data=np.array(test)
#print (train_data.shape)
#print (train_labels.shape)
#print (test_data.shape)
# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# Any files you write to the current directory get shown as outputs

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(train_data,train_labels.ravel())
out=[]
for i in test_data:
    p=int(classifier.predict(i))
    out.append(p)
    print (p)
print (out)