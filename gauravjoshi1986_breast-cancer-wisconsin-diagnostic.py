import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import some data to play with

df = pd.read_csv("../input/data.csv",header = 0)



df.head()
df = df.drop("Unnamed: 32",1)



X = df.iloc[:,2:] 

y = df.diagnosis
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

y = le.fit_transform(y)
from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline



pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=15)),('clf', LogisticRegression(random_state=1))])

pipe_lr.fit(X_train, y_train)



print (pipe_lr.named_steps['pca'].explained_variance_ratio_)

print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
import matplotlib.pyplot as plt

plt.plot(pipe_lr.named_steps['pca'].explained_variance_ratio_, linewidth=2)
import numpy as np

from sklearn.model_selection  import StratifiedKFold



skf = StratifiedKFold(n_splits=4, random_state=1)



scores = []

k = 0

#indicies = list(skf.split(X, y))



for train_index, test_index in skf.split(X, y):

    #print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]



    pipe_lr.fit(X_train, y_train)

    score = pipe_lr.score(X_test, y_test)

    scores.append(score)

    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y[train_index]), score))

    k = k + 1
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))