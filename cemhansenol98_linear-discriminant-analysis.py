from sklearn import datasets

import numpy as np



iris = datasets.load_iris()



X = iris.data

y = iris.target

import matplotlib.pyplot as plt



from sklearn import datasets

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



iris = datasets.load_iris()



X = iris.data

y = iris.target

target_names = iris.target_names





lda = LinearDiscriminantAnalysis(n_components=2)

X_r2 = lda.fit(X, y).transform(X)



colors = ['navy', 'turquoise', 'darkorange']



for color, i, target_name in zip(colors, [0, 1, 2], target_names):

    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,

                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('LDA of IRIS dataset')



plt.show()