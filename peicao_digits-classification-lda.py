import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

# Import the 3 dimensionality reduction methods
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
y_train = train.label
X_train = train.drop('label',axis = 1)
X_std = StandardScaler().fit_transform(X_train.values)
X_test = StandardScaler().fit_transform(test.values)

lda = LDA(n_components=3)
lda.fit(X_std, y_train.values)
y_test = lda.predict(X_test)
#print(y_test)

y_test = pd.Series(y_test, name = 'Label')
sub = pd.concat([pd.Series(range(1,28001),name = "ImageId"), y_test],axis = 1)
sub.to_csv("cnn_mnist_datagen.csv",index=False)
