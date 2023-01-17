import time

import statistics

import collections

import numpy

import pandas as pd

from IPython.display import display, FileLinks, HTML

from sklearn.metrics import accuracy_score, log_loss

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

from sklearn.svm import LinearSVC, SVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

import matplotlib.image
train = pd.read_csv('../input/train.csv')



species = train.pop('species')

le = LabelEncoder().fit(species.values)

y = le.transform(species)



ids = train.pop('id').values

x = train.values

scaler = StandardScaler().fit(x)

x = scaler.transform(x)



#x.shape, y.shape

ids
#plt.figure()

for i in range(99):

    id = ids[y == i][0]

    #display(id)

    img = matplotlib.image.imread('../input/images/{}.jpg'.format(id))

    plt.subplot(10, 10, i + 1)

    plt.title('imageID = ' + str(id))

    

#    display(ids[y == i])
start = time.time()



c = LogisticRegression(C=3000)#, penalty='l1')#, multi_class='multinomial', solver='sag')



scores = cross_val_score(c, x, y, cv=5, scoring='neg_log_loss')



display(scores)

display('Loss: %0.3f (+/- %0.3f)' % (scores.mean(), scores.std() * 2))



display('it took {}s'.format(time.time() - start))
c = c.fit(x, y)



test = pd.read_csv('../input/test.csv')

test_ids = test.pop('id')

x_test = test.values

x_test = scaler.transform(x_test)



y_test = c.predict_proba(x_test)



numpy.abs(c.coef_).sum(axis=0)#.reshape(3, 64)



#pd.DataFrame(y_test, index=test_ids, columns=le.classes_).to_csv('result.csv')

HTML('<pre>{}</pre>'.format(data))
import matplotlib.pyplot as plt

import matplotlib.image



for i in range(1, 100):

    img = matplotlib.image.imread('../input/images/{}.jpg'.format(i))

    _ = plt.figure()

    _ = plt.imshow(img)

plt.show()
