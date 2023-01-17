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
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

from sklearn.metrics import classification_report

from sklearn import metrics

from itertools import product

from sklearn import svm



import warnings

warnings.filterwarnings('ignore')





# Any results you write to the current directory are saved as output.
# Load datasets

# DataFrame 을 이용하면 편리하다.

data_table = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_train.csv')
train_data = data_table[['tempo', 'beats', 'chroma_stft','rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff','zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',

       'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12','mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19','mfcc20']]



train_labels = data_table [['label']]
scaler = StandardScaler()

train_data = train_data.values

scaler.fit(train_data)

train_data = scaler.transform(train_data)
train_labels = train_labels.values
labels = list()



for label in train_labels :

  if label == 'blues':

    labels.append(0)

  elif label == 'classical':

    labels.append(1)

  elif label == 'country':

    labels.append(2)

  elif label == 'disco':

    labels.append(3)

  elif label == 'hiphop':

    labels.append(4)

  elif label == 'jazz':

    labels.append(5)

  elif label == 'metal':

    labels.append(6)

  elif label == 'pop':

    labels.append(7)

  elif label == 'reggae':

    labels.append(8)

  else :

    labels.append(9)



labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.25, random_state=42)
from sklearn.model_selection import GridSearchCV

grid_P= {'C': [1, 10, 100, 1e3,1e4], 'gamma': [ 0.1,0.05,0.005, 0.0005] ,'class_weight' : ['balanced']}

svc = svm.SVC(kernel = 'rbf',gamma="scale")

clf = GridSearchCV(svc, grid_P, cv=10)

clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
# Load datasets

# DataFrame 을 이용하면 편리하다.

data_table = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_test.csv')



test_data = data_table [['tempo', 'beats', 'chroma_stft', 'rmse',

       'spectral_centroid', 'spectral_bandwidth', 'rolloff',

       'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',

       'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',

       'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',

       'mfcc20']]



test_data = test_data.values

scaler.fit(test_data)

test_data = scaler.transform(test_data)
result = clf.predict(test_data)

result = result.reshape(-1,1)

print(result.shape)
# numpy 를 Pandas 이용하여 결과 파일로 저장



import pandas as pd



print(result.shape)

df = pd.DataFrame(result, columns=["label"])

df.index = np.arange(1,len(df)+1)

df.index.name = 'id'

df.to_csv('results-hhnam.csv',index=True, header=True)