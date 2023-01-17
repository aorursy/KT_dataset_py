from IPython.core.display import display



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
data = pd.read_csv("../input/uci-news-aggregator.csv")

data.head()
X_raw, y = data['TITLE'], data['CATEGORY']
value_counts = dict(y.value_counts())

targets_labels = value_counts.keys()

ind = range(len(targets_labels))

plt.bar(ind, value_counts.values())

plt.title("Categories count")

plt.xticks(ind, targets_labels)

plt.show()
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(X_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = LinearSVC()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print (classification_report(y_test, y_pred))
def plot_confusion_matrix(y_true, y_pred, targets_labels=None):

    targets_labels = list(targets_labels)

    if not targets_labels:

        targets_labels = list(set(y_true))

    num_classes = len(targets_labels)

    cdata = confusion_matrix(y_true, y_pred, labels=targets_labels)

    cdata = cdata / cdata.sum(axis=1).astype(float)

    heatmap = plt.pcolor(cdata, cmap="PuBu")

    plt.title("Confusion matrix")

    plt.colorbar(heatmap)

    for y in range(cdata.shape[0]):

        for x in range(cdata.shape[1]):

            plt.text(x + 0.5, y + 0.5, '{0:.2f}%'.format((cdata[y, x] * 100)),

                     horizontalalignment='center',

                     verticalalignment='center',

                     )



    tick_marks = np.arange(num_classes) + 0.5

    plt.xticks(tick_marks, targets_labels)

    plt.yticks(tick_marks, targets_labels)

    plt.show()

    

plot_confusion_matrix(y_test, y_pred, targets_labels)