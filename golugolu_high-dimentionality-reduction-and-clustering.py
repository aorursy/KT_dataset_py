# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
label = train['label']

train = train.drop('label', axis = 1)
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# classifiers 

tsne = TSNE(n_components=3)

pca = PCA(n_components=3)

lda = LDA(n_components=3)
# transforming all the data

#tsne_result = tsne.fit_transform(train) 

pca_result = pca.fit_transform(train)

lda_result = lda.fit_transform(train,label)
import plotly.plotly as py

import plotly.graph_objs as go

x, y, z = pca_result[:100,0] , pca_result[:100,1] , pca_result[:100,2]

trace1 = go.Scatter3d(

    x=x,

    y=y,

    z=z

)

data = [trace1]

fig = go.Figure(data=data)

py.plot(fig)


