import numpy as np 

import pandas as pd 

from sklearn.preprocessing import StandardScaler

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from matplotlib.colors import ListedColormap

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

target = data.target

data = data.drop(['target'], axis = 'columns')
data.head(10)
scaler = StandardScaler()

data = scaler.fit_transform(data)
def pca_analysis(ncomps, data = data, target = target, plot = False):

    

    pca = PCA(n_components = ncomps)

    principal_comps = pca.fit_transform(data)

    principal_comps_df = pd.concat([pd.DataFrame(data = principal_comps, columns = ['pc' + str(i) for i in range(1, ncomps + 1)]), target], 

                                   axis = 'columns')

    

    if plot:

        if ncomps == 2:

            plt.figure(figsize = (8, 8))

            fig = sns.scatterplot(data = principal_comps_df, 

                                  x = 'pc1', 

                                  y = 'pc2', 

                                  hue = 'target')

            fig.set_title('2 Component PCA')

    return principal_comps_df
principal_comps_2d = pca_analysis(2, plot = False)
# Modified the code found on the following link to make this plot.

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py



h = .02  # step size in the mesh



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes", "QDA", "XGBoost"]



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    MLPClassifier(alpha=1, max_iter=1000),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(), 

    XGBClassifier()]



datasets = [(np.array(principal_comps_2d[['pc1', 'pc2']]), np.array(target))]



figure = plt.figure(figsize = (15, 15))

i = 1

# iterate over datasets

for ds_cnt, ds in enumerate(datasets):

    # preprocess dataset, split into training and test part

    X, y = ds

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5

    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))



    # just plot the dataset first

    cm = plt.cm.RdBu

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    ax = plt.subplot(4, 4, i)

    if ds_cnt == 0:

        ax.set_title("Input data")

    # Plot the training points

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,

               edgecolors='k')

    # Plot the testing points

    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,

               edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())

    ax.set_ylim(yy.min(), yy.max())

    ax.set_xticks(())

    ax.set_yticks(())

    i += 1



    # iterate over classifiers

    for name, clf in zip(names, classifiers):

        ax = plt.subplot(4, 4, i)

        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)



        # Plot the decision boundary. For that, we will assign a color to each

        # point in the mesh [x_min, x_max]x[y_min, y_max].

        if hasattr(clf, "decision_function"):

            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

        else:

            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]



        # Put the result into a color plot

        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)



        # Plot the training points

        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,

                   edgecolors='k')

        # Plot the testing points

        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,

                   edgecolors='k', alpha=0.6)



        ax.set_xlim(xx.min(), xx.max())

        ax.set_ylim(yy.min(), yy.max())

        ax.set_xticks(())

        ax.set_yticks(())

        if ds_cnt == 0:

            ax.set_title(name)

        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),

                size=15, horizontalalignment='right')

        i += 1



plt.tight_layout()

plt.show()