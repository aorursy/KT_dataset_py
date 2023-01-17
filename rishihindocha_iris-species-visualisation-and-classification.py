import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy.stats as stats

from sklearn import datasets, cross_validation, neighbors, svm

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
iris = datasets.load_iris()



df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],

                  columns = iris['feature_names'] + ['target'])



print("Number of rows: {}\n".format(len(df)))

print("Distribution of labels:\n{}".format(df['target'].value_counts(normalize=True)))

print("\nFeatures:")



df.drop(df.columns[4], axis=1).head()

# Drop the 'target' column as it contains the labels (0, 1 or 2). We only want to see the features.
for i in range (0, 4): # Loop through each feature column by index (0, 1, 2, 3).

    

    species_labels = [0, 1, 2]

    

    plt.rcParams["figure.figsize"] = [14, 5]

    ax = plt.subplot(2, 4, i+1) # Scatter graph on top.

    plt.scatter(df[df.columns[i]], df['target'], c=df['target'])

    if i is 0: plt.ylabel('species')

    plt.yticks(species_labels)

    

    plt.subplot(2, 4, i+5, sharex=ax) # Box plot below the scatter graph.

    cols = df[[df.columns[i], 'target']]

    data = [] # Group the data by species.

    for j in species_labels:

        data.append(cols[cols['target']==j][df.columns[i]])

    plt.boxplot(data, vert=False, labels=species_labels)

    plt.xlabel(df.columns[i])

    if i is 0: plt.ylabel('species')



plt.show()
def relationship_subplot(col1, col2, pos):

    plt.rcParams["figure.figsize"] = [14, 5]

    plt.subplot(1,2,pos)

    plt.scatter(df[df.columns[col1]], df[df.columns[col2]], c=df['target'])

    plt.xlabel(df.columns[col1])

    plt.ylabel(df.columns[col2])

    patches = []

    for number, colour in enumerate(['#390540', '#3C9586', '#F3E552']):

        patches.append(mpatches.Patch(color=colour, label=number))

    plt.legend(handles=patches) # Manually create a legend (do let me know if this can be automated).

    return stats.pearsonr(df[df.columns[col1]], df[df.columns[col2]])



petal = relationship_subplot(0,1,1)

sepal = relationship_subplot(2,3,2)



plt.show()



print("Petal Correlation (PCC, p-value): {}".format(petal))

print("Sepal Correlation (PCC, p-value): {}".format(sepal))
df.drop(df.columns[1], axis=1, inplace=True) # Remove sepal width.



X = df.drop(['target'], 1) # Features

y = df['target'] # Labels



X_train, X_test,y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)



print("There are {} rows of training data and {} rows of testing data.".format(len(X_train),

                                                                               len(X_test)))
classifier = VotingClassifier([('lsvc', svm.LinearSVC()),

                               ('knn', neighbors.KNeighborsClassifier()),

                               ('rfor', RandomForestClassifier())])



classifier.fit(X_train, y_train)

confidence = classifier.score(X_test, y_test)



print("Accuracy: {}".format(confidence))