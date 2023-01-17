import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

milk_powder = pd.read_csv('../input/milk-powder.csv')
milk_powder.head()
y = milk_powder.values[:, 1].astype('uint8')

X = milk_powder.values[:, 2:]



lda = LDA(n_components=2)

Xlda = lda.fit_transform(X, y)
import matplotlib.pyplot as plt
# Define the labels for the plot legend

labplot = [f'Milk {i*10}% ' for i in range(11)]
# Scatter plot

unique = list(set(y))
unique
import numpy as np

colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]

plt.figure(figsize=(10, 10))

with plt.style.context('ggplot'):

    for i, u in enumerate(unique):

        col = np.expand_dims(np.array(colors[i]), axis=0)

        xi = [Xlda[j, 0] for j in range(len(Xlda[:, 0])) if y[j] == u]

        yi = [Xlda[j, 1] for j in range(len(Xlda[:, 1])) if y[j] == u]

        plt.scatter(xi, yi, c = col, s=60, edgecolors='k', label=str(u))

    plt.xlabel('F1')

    plt.ylabel('F2')

    plt.legend(labplot, loc = 'lower right')

    plt.title('LDA')

    plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

# If you want it to be repeatable (for instance if you want to check the performance of the classifier on the same split by changing some other parameter)
lda = LDA()

lda.fit(X_train, y_train)

y_pred = lda.predict(X_test)

print(lda.score(X_test, y_test))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(LDA(), X, y, cv=4)
print(scores)

print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()*3))

print("Accuracy confidence interval (3*sigma) [%0.4f, %0.4f]" % (scores.mean()-scores.std()*3, min([scores.mean()+scores.std()*3, 1])))
from sklearn.decomposition import PCA
pca = PCA(n_components=15)

Xpc = pca.fit_transform(X)

scores = cross_val_score(LDA(), Xpc, y, cv = 4)
pca.explained_variance_ratio_
print(scores)

print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()*3))

print("Accuracy confidence interval (3*sigma) [%0.4f, %0.4f]" % (scores.mean()-scores.std()*3, min([scores.mean()+scores.std()*3, 1])))