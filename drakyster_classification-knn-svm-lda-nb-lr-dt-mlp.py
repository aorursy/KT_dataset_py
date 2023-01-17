import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv("../input/heart-disease-uci/heart.csv")



count_row = df.shape[0]  # gives number of row count

count_column = df.shape[1] # gives number of column count



print('Number of rows: {}'.format(count_row))

print('Number of columns: {}'.format(count_column))



df.head()
df.describe()
plt.figure(figsize=(10,10))

p=sns.heatmap(df.corr(), annot=True,cmap='RdYlGn',square=True, fmt='.2f')
# Check for missing data

df.isnull().sum() # gives number of null value by column
from sklearn.preprocessing import MinMaxScaler



X, y = df.drop(['target'], axis=1), df['target']



scaler = MinMaxScaler()

#scaler = StandardScaler()



# Optionnal

X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

X.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.neighbors import KNeighborsClassifier



n = 5



clf = KNeighborsClassifier(n)



clf.fit(X_train, y_train)



preds = clf.predict(X_test)

print('Test set predictions: {}'.format(preds))

print('\n')



#print('Train set accuracy: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))

#print('Test set accuracy: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Train set accuracy: {:.2f}'.format(clf.score(X_train, y_train)))

print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

print('\n')



confusion = confusion_matrix(y_test, preds)



print('Confusion matrix:\n', confusion)
training_accuracy = []

test_accuracy = []



n_range = range(1, 11)



for n in n_range:

    clf = KNeighborsClassifier(n)

    clf.fit(X_train, y_train)

    training_accuracy.append(clf.score(X_train, y_train))

    test_accuracy.append(clf.score(X_test, y_test))

    

plt.plot(n_range, training_accuracy, label='training accuracy')

plt.plot(n_range, test_accuracy, label='test accuracy')

plt.xlabel('n neighbors')

plt.ylabel('accuracy')

plt.legend()
from sklearn.svm import SVC



clf = SVC(kernel="linear", C=1)



clf.fit(X_train, y_train)



preds = clf.predict(X_test)

#print('Test set predictions: {}'.format(preds))

#print('\n')



#print('Train set accuracy: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))

#print('Test set accuracy: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Train set accuracy: {:.2f}'.format(clf.score(X_train, y_train)))

print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

print('\n')



confusion = confusion_matrix(y_test, preds)



print('Confusion matrix:\n', confusion)



plt.bar(range(X_train.shape[1]), clf.coef_[0], align = 'center', alpha = 0.5)

plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)

xlims = plt.xlim()

plt.hlines(0, xlims[0], xlims[1])
training_accuracy = []

test_accuracy = []



C_range = np.logspace(-3, 3, num=7)



for C in C_range:

    clf = SVC(kernel="linear", C=C)

    clf.fit(X_train, y_train)

    training_accuracy.append(clf.score(X_train, y_train))

    test_accuracy.append(clf.score(X_test, y_test))

    

plt.plot(C_range, training_accuracy, label='training accuracy')

plt.plot(C_range, test_accuracy, label='test accuracy')

plt.xlabel('C')

plt.xscale("log")

plt.ylabel('accuracy')

plt.legend()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



clf = LinearDiscriminantAnalysis()



clf.fit(X_train, y_train)



preds = clf.predict(X_test)

#print('Test set predictions: {}'.format(preds))

#print('\n')



#print('Train set accuracy: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))

#print('Test set accuracy: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Train set accuracy: {:.2f}'.format(clf.score(X_train, y_train)))

print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

print('\n')



confusion = confusion_matrix(y_test, preds)



print('Confusion matrix:\n', confusion)
from sklearn.naive_bayes import GaussianNB



clf = GaussianNB()



clf.fit(X_train, y_train)



preds = clf.predict(X_test)

#print('Test set predictions: {}'.format(preds))

#print('\n')



#print('Train set accuracy: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))

#print('Test set accuracy: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Train set accuracy: {:.2f}'.format(clf.score(X_train, y_train)))

print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

print('\n')



confusion = confusion_matrix(y_test, preds)



print('Confusion matrix:\n', confusion)
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(C=1)



clf.fit(X_train, y_train)



preds = clf.predict(X_test)

#print('Test set predictions: {}'.format(preds))

#print('\n')



#print('Train set accuracy: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))

#print('Test set accuracy: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Train set accuracy: {:.2f}'.format(clf.score(X_train, y_train)))

print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

print('\n')



confusion = confusion_matrix(y_test, preds)



print('Confusion matrix:\n', confusion)



plt.bar(range(X_train.shape[1]), clf.coef_[0], align = 'center', alpha = 0.5)

plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)

xlims = plt.xlim()

plt.hlines(0, xlims[0], xlims[1])
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=11)



clf.fit(X_train, y_train)



preds = clf.predict(X_test)

#print('Test set predictions: {}'.format(preds))

#print('\n')



#print('Train set accuracy: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))

#print('Test set accuracy: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Train set accuracy: {:.2f}'.format(clf.score(X_train, y_train)))

print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

print('\n')



confusion = confusion_matrix(y_test, preds)



print('Confusion matrix:\n', confusion)
training_accuracy = []

test_accuracy = []



max_depth_range = range(1, 10)



for max_depth in max_depth_range:

    clf = DecisionTreeClassifier(max_depth=max_depth)

    clf.fit(X_train, y_train)

    training_accuracy.append(clf.score(X_train, y_train))

    test_accuracy.append(clf.score(X_test, y_test))

    

plt.plot(max_depth_range, training_accuracy, label='training accuracy')

plt.plot(max_depth_range, test_accuracy, label='test accuracy')

plt.xlabel('max depth')

plt.ylabel('accuracy')

plt.legend()
training_accuracy = []

test_accuracy = []



min_samples_leaf_range = range(15, 1, -1)



for min_samples_leaf in min_samples_leaf_range:

    clf = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)

    clf.fit(X_train, y_train)

    training_accuracy.append(clf.score(X_train, y_train))

    test_accuracy.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()



plt.plot(min_samples_leaf_range, training_accuracy, label='training accuracy')

plt.plot(min_samples_leaf_range, test_accuracy, label='test accuracy')

plt.xlabel('min samples leaf')

plt.ylabel('accuracy')

ax.set_xlim(15, 0)

plt.legend()
from sklearn.tree import export_graphviz



export_graphviz(clf, out_file='tree.dot', class_names=['0', '1'], feature_names=df.columns[:-1], impurity=False, filled=True)
import graphviz



with open('tree.dot') as f:

    dot_graph = f.read()

display(graphviz.Source(dot_graph))
def plot_feature_importances(model):

    n_features = X_train.shape[1]

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), X_train.columns)

    plt.xlabel('Feature importance')

    plt.ylabel('Feature')

    plt.ylim(-1, n_features)

    

plot_feature_importances(clf)
from sklearn.linear_model import Perceptron



clf = Perceptron(tol=1e-3)



clf.fit(X_train, y_train)



preds = clf.predict(X_test)

#print('Test set predictions: {}'.format(preds))

#print('\n')



#print('Train set accuracy: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))

#print('Test set accuracy: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Train set accuracy: {:.2f}'.format(clf.score(X_train, y_train)))

print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

print('\n')



confusion = confusion_matrix(y_test, preds)



print('Confusion matrix:\n', confusion)
class MyPerceptron():  

    """My own implementation of the perceptron"""



    def __init__(self, n_epochs=5, lr=0.01):

        """

        Called when initializing the classifier

        """

        self.n_epochs = n_epochs

        self.lr = lr



    def fit(self, X, Y=None):

        """

        Fit classifier

        """

        assert (type(self.n_epochs) == int), "n_epochs parameter must be integer"

        assert (type(self.lr) == float), "lr parameter must be float"



        self.weights = np.random.rand(X.shape[1] + 1)

        

        for epoch in range(self.n_epochs):

            for x, y in zip(X.values, Y.values):

                pred = self.predict(x)

                self.weights[1:] += self.lr * (y - pred) * x

                self.weights[0] += self.lr * (y - pred)



        return self



    def predict(self, X, Y=None):

        """

        Make predictions

        """

        try:

            getattr(self, "weights")

        except AttributeError:

            raise RuntimeError("You must train classifer before making predictions!")



        return(np.heaviside(np.dot(X, self.weights[1:]) + self.weights[0], 1))



    def score(self, X, Y=None):

        """

        Compute accuracy score

        """

        return(accuracy_score(Y, self.predict(X))) 
clf = MyPerceptron()



clf.fit(X_train, y_train)



preds = clf.predict(X_test)

#print('Test set predictions: {}'.format(preds))

#print('\n')



#print('Train set accuracy: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))

#print('Test set accuracy: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Train set accuracy: {:.2f}'.format(clf.score(X_train, y_train)))

print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

print('\n')



confusion = confusion_matrix(y_test, preds)



print('Confusion matrix:\n', confusion)
training_accuracy = []

test_accuracy = []



n_epochs_range = range(1, 50)



for n_epochs in n_epochs_range:

    clf = MyPerceptron(n_epochs=n_epochs)

    clf.fit(X_train, y_train)

    training_accuracy.append(clf.score(X_train, y_train))

    test_accuracy.append(clf.score(X_test, y_test))



fig, ax = plt.subplots()



plt.plot(n_epochs_range, training_accuracy, label='training accuracy')

plt.plot(n_epochs_range, test_accuracy, label='test accuracy')

plt.xlabel('n epochs')

plt.ylabel('accuracy')

plt.legend()
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(clf, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())