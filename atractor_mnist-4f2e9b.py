import pandas as pd



# The competition datafiles are in the directory ../input

# Read competition data files:

train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")



# Write to the log:

print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# Any files you write to the current directory get shown as outputs





import numpy as np
def trim_images(images):

    cols = set(['pixel' + str(i + j * 28) for i in [0,1,2,3, 24,25,26,27] for j in range(0, 28)])

    cols |= set(['pixel' + str(j + i * 28) for i in [0,1,2,3, 24,25,26,27] for j in range(0, 28)])

    return images.drop(list(cols), 1)

images = pd.DataFrame(trim_images(train.drop('label', 1)))

labels = train['label']
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(bootstrap=False, class_weight='auto', criterion='gini',

            max_depth=None, max_features=0.032, max_leaf_nodes=None,

            min_samples_leaf=1, min_samples_split=1,

            min_weight_fraction_leaf=0.0, n_estimators=140, n_jobs=8,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)



n_train = int(len(images) * 0.75)



clf.fit(images[:n_train], labels[:n_train])



len(images[n_train:][clf.predict(images[n_train:]) == labels[n_train:]]) / (len(images) - n_train)
predication = pd.DataFrame(clf.predict(trim_images(test)), columns=['Label'], index=(test.index + 1))

predication.to_csv('output.csv', index=True, index_label='ImageId')