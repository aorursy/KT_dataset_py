from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import numpy

from tqdm import tqdm

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from sklearn.model_selection import KFold



x,y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant= 0, n_clusters_per_class=1, random_state=60)

X_train, X_test, y_train, y_test = train_test_split(x,y,stratify=y,random_state=42)
%matplotlib inline

import matplotlib.pyplot as plt

colors = {0:'red', 1:'blue'}

plt.scatter(X_test[:,0], X_test[:,1],c=y_test)

plt.show()
import numpy as np

import pandas as pd

def RandomSearchCV(x_train,y_train,classifier, param_range, folds):

  params = np.random.uniform(param_range[0],param_range[1],10)

  params = np.array([int(i) for i in params])

  params = np.sort(params)

  # Reference link : https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f

  kf = KFold(n_splits=folds)



  x_train = pd.DataFrame(x_train)

  y_train = pd.DataFrame(y_train)



  TRAIN_SCORES = []

  TEST_SCORES  = [] 

  for p in params:



    training_scores = []

    crossval_scores = []

    classifier.n_neighbors = int(p)



    for i in range(folds):

      result = next(kf.split(x_train),None)

      x_training = x_train.iloc[result[0]]

      x_cv = x_train.iloc[result[1]]



      y_training = y_train.iloc[result[0]]

      y_cv = y_train.iloc[result[1]]

      

      model = classifier.fit(x_training,y_training)

      training_scores.append(model.score(x_training,y_training))

      crossval_scores.append(model.score(x_cv,y_cv))

    TRAIN_SCORES.append(np.mean(training_scores))

    TEST_SCORES.append(np.mean(crossval_scores))

  return(TRAIN_SCORES , TEST_SCORES)

     

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



classifier = KNeighborsClassifier()

X_train = pd.DataFrame(X_train)

y_train = pd.DataFrame(y_train)

train_score , cv_scores = RandomSearchCV(X_train,y_train,classifier,(1,21),8)
# 6. plot hyper-parameter vs accuracy plot as shown in reference notebook and choose the best hyperparameter

import matplotlib.pyplot as plt

params = np.random.uniform(1,21,10)

params = np.array([int(i) for i in params])

params = np.sort(params)

#params = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

plt.plot(params,train_score, label='train cruve')

plt.plot(params,cv_scores, label='test cruve')

plt.xlabel("Hyperparameter k")

plt.ylabel("Accuracy")

plt.title('Hyper-parameter VS accuracy plot')

plt.legend()

plt.show()
from matplotlib.colors import ListedColormap
neigh = KNeighborsClassifier(n_neighbors = 13)

neigh.fit(X_train, y_train)
X1 = np.array(X_train[0])

X2 = np.array(X_train[1])

y = np.array(y_train)

y = [j for sub in y for j in sub]
# understanding this code line by line is not that importent 

def plot_decision_boundary(X1, X2, y, clf):

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



    x_min, x_max = X1.min() - 1, X1.max() + 1

    y_min, y_max = X2.min() - 1, X2.max() + 1

    

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)



    plt.figure()

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points

    plt.scatter(X1, X2, c=y, cmap=cmap_bold)

    

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())

    plt.title("2-Class classification (k = %i)" % (clf.n_neighbors))

    plt.show()
plot_decision_boundary(X1,X2,y, neigh)