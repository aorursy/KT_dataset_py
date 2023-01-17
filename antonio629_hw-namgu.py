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

#from pandas.tools.plotting import scatter_matrix

from pandas.plotting import autocorrelation_plot



import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from mpl_toolkits.mplot3d import axes3d, Axes3D

import seaborn as sns



from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedShuffleSplit



from sklearn.svm import SVC

from sklearn.neighbors import NearestCentroid

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import metrics



from itertools import product



import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import scale

from sklearn.pipeline import make_pipeline

df_data = pd.read_csv('/content/input2/data_train.csv')

df = df_data

df.head()

X = df_data[['tempo', 'beats', 'chroma_stft', 'rmse',

       'spectral_centroid', 'spectral_bandwidth', 'rolloff',

       'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',

       'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',

       'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',

       'mfcc20']]

X = scale(X)



y = df_data[['label']]

y = np.array(y)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
pca = PCA(n_components=None, whiten=True, random_state=42)

svc = SVC(kernel='rbf', class_weight='balanced')

model = make_pipeline(pca, svc)



param_grid = {'svc__C': [1, 5, 10,15,20,25,30],

              'svc__gamma': [0.01, 0.015,0.02,0.025,0.03]}



grid = GridSearchCV(model, param_grid)

%time grid.fit(X_train, y_train)

print(grid.best_params_)

df_data_test = pd.read_csv('/content/input2/data_test.csv')

df = df_data_test

df.head()





X_test = df_data_test[['tempo', 'beats', 'chroma_stft', 'rmse',

       'spectral_centroid', 'spectral_bandwidth', 'rolloff',

       'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',

       'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',

       'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',

       'mfcc20']]

X_test = scale(X_test)



y_test = np.array(df_data_test[['label']])

model = grid.best_estimator_

result = model.predict(X_test)
import pandas as pd



print(result.shape)

df = pd.DataFrame(data=result, index=range(1,51), columns=['label'])

df = df.replace('blues',0)

df = df.replace('classical',1)

df = df.replace('country',2)

df = df.replace('disco',3)

df = df.replace('hiphop',4)

df = df.replace('jazz',5)

df = df.replace('metal',6)

df = df.replace('pop',7)

df = df.replace('reggae',8)

df = df.replace('rock',9)



df.to_csv('results-nk-v2.csv',index=True, header=True, index_label='id')