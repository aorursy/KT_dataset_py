# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import numpy as np

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the train and test dataset, which is in the "../input/" directory
train = pd.read_csv("../input/train.csv") # the train dataset is now a Pandas DataFrame
test = pd.read_csv("../input/test.csv") # the train dataset is now a Pandas DataFrame

# Let's see what's in the trainings data - Jupyter notebooks print the result of the last thing you do
train.head()
X = train.iloc[:,:-1]
y = train.TARGET

X['n0'] = (X==0).sum(axis=1)
train['n0'] = X['n0']
Xs = X[['var15','var38','saldo_var30','n0']]
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(Xs, y)
from IPython.display import Image
from sklearn.externals.six import StringIO  
import pydot
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=Xs.columns,  
                         class_names=['Happy','Unhappy'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  
# adapted from http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html#example-tree-plot-iris-py
plot_step = 0.02
n_classes = 2
plot_colors = "bry"
plt.figure(figsize=(16,8))
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    Xp = Xs.ix[:,pair]
    yp = y

    # Shuffle
    idx = np.arange(Xp.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    Xp = Xp.loc[idx]
    yp = yp.loc[idx]

    # Standardize
    mean = Xp.mean(axis=0)
    std = Xp.std(axis=0)
    Xp = (Xp - mean) / std

    # Train
    clf = tree.DecisionTreeClassifier().fit(Xp, yp)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = Xp.ix[:, 0].min() - 1, Xp.ix[:, 0].max() + 1
    y_min, y_max = Xp.ix[:, 1].min() - 1, Xp.ix[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(Xs.columns[pair[0]])
    plt.ylabel(Xs.columns[pair[1]])
    plt.axis("tight")

    labels = ['unhappy', 'happy']
    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        plt.scatter(Xp.ix[y==i,0], Xp.ix[y==i,1], c=color, label=labels[i],
                    cmap=plt.cm.Paired)

    plt.axis("tight")

plt.suptitle("Decision surface of a decision tree using paired features")
#plt.legend()
plt.savefig('decision_surfaces.png', bbox_inches='tight', pad_inches=1)
plt.show()
