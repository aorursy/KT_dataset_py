# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
colors = ['royalblue','red','deeppink', 'maroon', 'mediumorchid', 'tan', 'forestgreen', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
lda.explained_variance_ratio_
plt.scatter(X_r2[:,0],X_r2[:,1],c=vectorizer(y))
pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)
from pylab import *
subplot(2,1,1)
title("PCA")
plt.scatter(X_r[:,0],X_r[:,1],c=vectorizer(y))
subplot(2,1,2)
title("LDA")
plt.scatter(X_r2[:,0],X_r2[:,1],c=vectorizer(y))
import seaborn as sns
df=pd.DataFrame(zip(X_r[:,0],X_r[:,1],X_r2[:,0],X_r2[:,1],y),columns=["pc1","pc2","ld1","ld2","class"])
subplot(2,1,1)
sns.boxplot(x='class', y='ld1', data=df)
subplot(2,1,2)
sns.boxplot(x='class', y='ld2', data=df)
subplot(2,1,1)
sns.boxplot(x='class', y='ld1', data=df)
subplot(2,1,2)
sns.boxplot(x='class', y='pc1', data=df)
pc.columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train, y_train)
#x_test_r2=lda.transform(X_test)
from sklearn.metrics import accuracy_score
y_pred = lda.predict(X_test)
print(accuracy_score(y_test, y_pred))
mnist_train=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")
mnist_test=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")
mnist_test.head(1)
y_train=mnist_train.iloc[:,0]
X_train=mnist_train.iloc[:,1:785]
lda = LinearDiscriminantAnalysis(n_components=9)
X_train_r2 = lda.fit(X_train, y_train).transform(X_train)
X_train_r2
lda.explained_variance_ratio_
subplot(1,2,1)
scatter=plt.scatter(X_train.iloc[:,200],X_train.iloc[:,320],c=y_train,cmap="Spectral")
handles, labels = scatter.legend_elements()
plt.legend(handles, labels,loc=0)
# Print out labels to see which appears first
subplot(1,2,2)
plt.scatter(X_train_r2[:,0],X_train_r2[:,1],c=y_train,cmap="Spectral")
handles, labels = scatter.legend_elements()
plt.legend(handles, labels)
subplot(1,2,1)
plt.scatter(X_train_r2[:,7],X_train_r2[:,8],c=y_train,cmap="Spectral")
handles, labels = scatter.legend_elements()
plt.legend(handles, labels)
# Print out labels to see which appears first
subplot(1,2,2)
plt.scatter(X_train_r2[:,0],X_train_r2[:,1],c=y_train,cmap="Spectral")
handles, labels = scatter.legend_elements()
plt.legend(handles, labels)
y_test=mnist_test.iloc[:,0]
X_test=mnist_test.iloc[:,1:785]
from sklearn.metrics import accuracy_score
y_pred = lda.predict(X_test)
print(accuracy_score(y_test, y_pred))
iris = datasets.load_iris()
X = iris.data
y = iris.target
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)
from pylab import *
subplot(1,3,1)
title("PCA")
plt.scatter(X_r[:,0],X_r[:,1],c=vectorizer(y))
subplot(1,3,2)
title("LDA")
plt.scatter(X_r2[:,0],X_r2[:,1],c=vectorizer(y))
subplot(1,3,3)
title("LDA and PCA")
plt.scatter(X_r[:,0],X_r2[:,0],c=vectorizer(y))