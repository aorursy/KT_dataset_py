!pip install diffprivlib



import diffprivlib.models as dp

import numpy as np

from sklearn.linear_model import LogisticRegression



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler



from sklearn.metrics import accuracy_score
X_train = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",

                        usecols=(0, 4, 10, 11, 12), delimiter=", ") # only use integer features for training



y_train = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",

                        usecols=14, dtype=str, delimiter=", ") # set last column as prediction label

np.unique(y_train)
X_test = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",

                        usecols=(0, 4, 10, 11, 12), delimiter=", ", skiprows=1)



y_test = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",

                        usecols=14, dtype=str, delimiter=", ", skiprows=1)



y_test = np.array([a[:-1] for a in y_test]) # Must trim trailing period "." from label



np.unique(y_test)
logreg = Pipeline([

    ('scaler', MinMaxScaler()), # scale min max of features to better fit sigmoid curve, control norm of data

    ('clf', LogisticRegression(solver="lbfgs")) # this process is smoothen by using pipeline in sklearn

])
logreg.fit(X_train, y_train)



print("Vanilla test accuracy: %.2f%%" % (accuracy_score(y_test, logreg.predict(X_test)) * 100))


dp_logreg = Pipeline([

    ('scaler', MinMaxScaler()),

    ('clf', dp.LogisticRegression())

])



dp_logreg.fit(X_train, y_train)



print("Differentially private test accuracy (epsilon=%.2f): %.2f%%" % 

     (dp_logreg['clf'].epsilon, accuracy_score(y_test, dp_logreg.predict(X_test)) * 100))
dp_logreg = Pipeline([

    ('scaler', MinMaxScaler()),

    ('clf', dp.LogisticRegression(epsilon=float("inf"), data_norm=2))

])



dp_logreg.fit(X_train, y_train)



print("Similarity between vanilla and differentially private (epsilon=inf) classifiers: %.2f%%" % 

     (accuracy_score(logreg.predict(X_test), dp_logreg.predict(X_test)) * 100))
accuracy = [accuracy_score(y_test, logreg.predict(X_test))]

epsilons = np.logspace(-3, 1, 500)



for eps in epsilons:

    dp_logreg.set_params(clf__epsilon=eps).fit(X_train, y_train)

    accuracy.append(accuracy_score(y_test, dp_logreg.predict(X_test)))
import pickle # result is saved using pickle so as to be use for plotting graph

import matplotlib.pyplot as plt



pickle.dump((epsilons, accuracy), open("logreg_accuracy_500.p", "wb" ) )



epsilons, accuracy = pickle.load(open("logreg_accuracy_500.p", "rb"))



plt.semilogx(epsilons, accuracy[1:], label="Differentially private")

plt.plot(epsilons, np.ones_like(epsilons) * accuracy[0], dashes=[2,2], label="Vanilla")

plt.title("Differentially private logistic regression accuracy")

plt.xlabel("epsilon")

plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.xlim(epsilons[0], epsilons[-1])

plt.legend(loc=4)

plt.show()