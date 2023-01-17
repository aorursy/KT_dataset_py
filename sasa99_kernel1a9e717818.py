# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
# save the labels to a Pandas series target

train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
target = train['label']

# Drop the label feature

train = train.drop("label",axis=1)

train.shape
from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
def train_using_pca(n_comp):

    

    # spliting

    

    x_train, x_test, y_train, y_test = train_test_split( train, target, test_size=0.25, random_state=42)

    

    # doing pca

    

    n_components = n_comp

    print(x_train.values.shape)

    pca = PCA(n_components=n_components).fit(x_train.values)

    eigenvalues = pca.components_.reshape(n_components, 28, 28)

    eigenvalues = pca.components_

    

    # making a new data frame with fewer dimensions

    x_train = DataFrame(pca.transform(x_train), index=x_train.index)

    

    # scaling 

    scaler = StandardScaler()

    x_train = DataFrame(scaler.fit_transform(x_train))

    

    

    #training

    

    model = SVC(kernel='rbf')

    model.fit(x_train, y_train)

    

    #seeing results

    x_test = pca.transform(x_test)

    x_test = DataFrame(scaler.transform(x_test))

    

    print("On train: ", accuracy_score(y_train, model.predict(x_train), normalize=True) )

    print("On test: ", accuracy_score(y_test, model.predict(x_test), normalize=True))

    

    return pca, scaler, model
pca, scaler, model = train_using_pca(20)
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
ids = test['id']

test = test.drop('id', axis=1)

test_pca = DataFrame(pca.transform(test))

test_pca = DataFrame(scaler.transform(test_pca))

preds = model.predict(test_pca)
output = pd.DataFrame({'id': ids, 'label': preds})

output.to_csv('submission.csv', index=False)