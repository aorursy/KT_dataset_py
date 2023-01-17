import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import to_categorical
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data
!head -n 5 promoters.data
df = pd.read_csv('promoters.data', names=['label', 'read_name', 'read'])
df.head()
df['read'] = df['read'].apply(lambda x: x.strip('\t'))
df.head()
def one_hot_encoding(seq):
    mp = dict(zip('acgt', range(4)))
    seq_2_number = [mp[nucleotide] for nucleotide in seq]
    return to_categorical(seq_2_number, num_classes=4, dtype='float32').flatten()
one_hot_encoding('gcata')
df['read'] = df['read'].apply(lambda seq: one_hot_encoding(seq))
df['label'] = df['label'].apply(lambda l: {'-' : 0, '+' : 1}[l])
df.head()
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(df['read'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
names = ["Nearest Neighbors", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "SVM Linear", "SVM RBF", "SVM Sigmoid"]

classifiers = [
    KNeighborsClassifier(n_neighbors = 3),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    SVC(kernel = 'linear'), 
    SVC(kernel = 'rbf'),
    SVC(kernel = 'sigmoid')
]


for name, classifier in zip(names, classifiers):
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(name, ': ' , accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))