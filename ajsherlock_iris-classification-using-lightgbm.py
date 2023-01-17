import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets

from sklearn.decomposition import PCA

import numpy as np

import lightgbm as lgb



# import some data to play with

iris = datasets.load_iris()

X = iris.data[:, :2]  # we only take the first two features.

y = iris.target



x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5

y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5



plt.figure(2, figsize=(8, 6))

plt.clf()



# Plot the training points

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,

            edgecolor='k')

plt.xlabel('Sepal length')

plt.ylabel('Sepal width')



plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

plt.xticks(())

plt.yticks(())



# To getter a better understanding of interaction of the dimensions

# plot the first three PCA dimensions

fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,

           cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])



plt.show()
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

X = iris.data  # we only take the first two features.

y = iris.target

print(np.shape(X))

print(np.shape(y))

le = preprocessing.LabelEncoder() #

y_label=le.fit_transform(y)

classes=le.classes_
X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.30, random_state=42)
params = {

          "objective" : "multiclass",

          "num_class" : 4,

          "num_leaves" : 60,

          "max_depth": -1,

          "learning_rate" : 0.01,

          "bagging_fraction" : 0.9,  # subsample

          "feature_fraction" : 0.9,  # colsample_bytree

          "bagging_freq" : 5,        # subsample_freq

          "bagging_seed" : 2018,

          "verbosity" : -1 }
lgtrain, lgval = lgb.Dataset(X_train, y_train), lgb.Dataset(X_test, y_test)

lgbmodel = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgval], early_stopping_rounds=100, verbose_eval=200)
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels



y_pred =np.argmax(lgbmodel.predict(X_test),axis=1)

y_true =y_test
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
plot_confusion_matrix(y_true, y_pred, classes=classes,

                      title='Confusion matrix, without normalization')
from sklearn.metrics import accuracy_score

accuracy_score(y_true, y_pred)