import numpy as np

import numpy.linalg

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['image.cmap'] = 'gray'



import time

from IPython.display import HTML, display

import tabulate
def read_vectors(*filenames):

    data = np.vstack(

        tuple(np.fromfile(filename, dtype=np.uint8).reshape(-1,401)

                      for filename in filenames))

    return data[:,1:], data[:,0] - 1



td1 = time.time()

X_train, y_train = read_vectors(*[

    "../input/snake-eyes/snakeeyes_{:02d}.dat".format(nn) for nn in range(10)])

X_test, y_test = read_vectors("../input/snake-eyes/snakeeyes_test.dat")

td2 = time.time()

print("Loading the data took {:.2f}s".format(td2-td1))
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier



def test_scikit_knn(Nsamples):

    t1 = time.time()

    neigh = KNeighborsClassifier(n_neighbors=5)

    neigh.fit(X_train[:Nsamples], y_train[:Nsamples])

    t2 = time.time()

    y_hat = neigh.predict_proba(X_test[:1000])

    t3 = time.time()

    llo = metrics.log_loss(np.eye(12)[y_test[:1000]], y_hat)

    acc = metrics.accuracy_score(y_test[:1000], np.argmax(y_hat, axis=1))

    return (Nsamples, acc, llo, t2-t1, t3-t2)
sk_table = [test_scikit_knn(N) for N in [20000, 40000, 80000]]
display(HTML(tabulate.tabulate(sk_table, ["Nsamples", "Accuracy", "Log-loss", "Training time", "Test time"], tablefmt='html')))
import annoy



def test_annoy_knn(Ntrees, Nsamples):

    t1 = time.time()

    

    vector_length = 400

    t = annoy.AnnoyIndex(vector_length)

    for i, v in zip(range(Nsamples), X_train[:Nsamples]):

        t.add_item(i, v)

    t.build(Ntrees)



    t2 = time.time()

    y_hat = [y_train[t.get_nns_by_vector(v, 1)[0]] for v in X_test]

    t3 = time.time()

    acc = metrics.accuracy_score(y_test, y_hat)

    conf = metrics.confusion_matrix(y_test, y_hat)

    return (Nsamples, acc, t2-t1, t3-t2, conf)
ann_results = [test_annoy_knn(nt, N) for nt in [1,10] for N in [20000, 40000, 80000, 1000000]]
ann_table = [x[:-1] for x in ann_results]

display(HTML(tabulate.tabulate(ann_table[:4], ["Nsamples", "Accuracy", "Training time", "Test time"], tablefmt='html')))

display(HTML(tabulate.tabulate(ann_table[4:], ["Nsamples", "Accuracy", "Training time", "Test time"], tablefmt='html')))
for x in ann_results:

    print(x[-1])
vector_length = 400

tbig = annoy.AnnoyIndex(vector_length)

for i, v in zip(range(1000000), X_train):

    tbig.add_item(i, v)

tbig.build(10)
from sklearn import metrics

from sklearn import linear_model

from sklearn import preprocessing

from sklearn import svm



def get_proba_from_points(X_t, y_t, X):

    if sum(y_t!=y_t[0]) < 1:

        out = np.zeros(12)

        out[y_t[0]] = 1

        return out

    clf = linear_model.LogisticRegression()

    clf.fit(X_t, y_t)    

    out = np.zeros(12)

    out[clf.classes_] = clf.predict_proba([X])

    return out



def test_annoy_knn_lda():

    t2 = time.time()

    query = [(v, np.array(tbig.get_nns_by_vector(v, 51))) for v in X_test]

    y_hat = [get_proba_from_points(

        X_train[q],

        y_train[q], v) for v,q in query]

    t3 = time.time()

    llo = metrics.log_loss(np.eye(12)[y_test], y_hat)

    acc = metrics.accuracy_score(y_test, np.argmax(y_hat, axis=1))

    conf = metrics.confusion_matrix(y_test, np.argmax(y_hat, axis=1))



    return (acc, llo, t3-t2, conf)
annlda_results = test_annoy_knn_lda()
annlda_table = [annlda_results[:-1]]

display(HTML(tabulate.tabulate(annlda_table, ["Accuracy", "Log-loss", "Test time"], tablefmt='html')))
print(annlda_results[-1])

plt.figure(figsize=(10,10))

plt.subplot(1,3,1)

plt.imshow(annlda_results[-1])

plt.subplot(1,3,2)

plt.imshow(ann_results[-1][-1])

plt.subplot(1,3,3)

plt.imshow(np.array(ann_results[-1][-1]) - np.array(annlda_results[-1]))