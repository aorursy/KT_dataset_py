import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline   

plt.rcParams['image.cmap'] = 'gray'



def read_vectors(filename):

    return np.fromfile(filename, dtype=np.uint8).reshape(-1,401)
snk = np.vstack(tuple(read_vectors("../input/snake-eyes/snakeeyes_{:02d}.dat".format(nn))

                      for nn in range(2)))

snk_y = snk[:,0]

snk_X = snk[:,1:]



plt.figure(figsize=(10,5))

plt.suptitle("Let's roll the bones")

for n in range(40):

    plt.subplot(4,10,n+1)

    plt.title(snk_y[n])

    plt.imshow(snk_X[n].reshape(20,20))

    plt.axis('off')
plt.hist(snk_y, range(14))
# Import datasets, classifiers and performance metrics

from sklearn import metrics

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# Create a classifier: LDA

clf = LinearDiscriminantAnalysis()



# We learn the digits on the first half of the digits

clf.fit(snk_X, snk_y)



test = read_vectors("../input/snake-eyes/snakeeyes_test.dat")

test_y = test[:,0]

test_X = test[:,1:]



expected = test_y

predicted = clf.predict(test_X)



print("Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected, predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))