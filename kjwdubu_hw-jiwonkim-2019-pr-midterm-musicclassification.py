ls -al
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



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

from sklearn import svm



import warnings

warnings.filterwarnings('ignore')



scaler = StandardScaler()

# Any results you write to the current directory are saved as output.
# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv("../input/2019-pr-midterm-musicclassification/data_train.csv")

                      

train_data = df_data [['tempo', 'beats', 'chroma_stft', 'rmse',\

       'spectral_centroid', 'spectral_bandwidth', 'rolloff',\

       'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',\

       'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',\

       'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',\

       'mfcc20']]



train_labels = df_data [['label']]



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

X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.05, random_state=42)
parameters = {'kernel':['rbf'], 'C': [1,5,10,50,100,50,1000], 'gamma': [ 0.1,0.05,0.01,0.005,0.001,0.0005,0.0001] ,'class_weight' : ['balanced']}

svc = svm.SVC(gamma="scale")

clf = GridSearchCV(svc, parameters, cv=5)

clf.fit(X_train, y_train)
clf.best_params_

y_predict = clf.predict(X_test)

confusion_matrix(y_test, y_predict)

print(classification_report(y_test, y_predict))
# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv("../input/2019-pr-midterm-musicclassification/data_test.csv")



test_data = df_data [['tempo', 'beats', 'chroma_stft', 'rmse',

       'spectral_centroid', 'spectral_bandwidth', 'rolloff',

       'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',

       'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',

       'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',

       'mfcc20']]



test_labels = df_data [['label']]



test_labels = test_labels.values



test_data = test_data.values

scaler.fit(test_data)

test_data = scaler.transform(test_data)
test_label = list()



for label in test_labels :

  if label == 'blues':

    test_label.append(0)

  elif label == 'classical':

    test_label.append(1)

  elif label == 'country':

    test_label.append(2)

  elif label == 'disco':

    test_label.append(3)

  elif label == 'hiphop':

    test_label.append(4)

  elif label == 'jazz':

    test_label.append(5)

  elif label == 'metal':

    test_label.append(6)

  elif label == 'pop':

    test_label.append(7)

  elif label == 'reggae':

    test_label.append(8)

  else :

    test_label.append(9)



test_label = np.array(test_label)
result = clf.predict(test_data)

result = result.reshape(-1,1)

print(result.shape)

print(classification_report(result, test_label))
# numpy 를 Pandas 이용하여 결과 파일로 저장



import pandas as pd



print(result.shape)

df = pd.DataFrame(result, columns=["label"])

df.index = np.arange(1,len(df)+1)

df.index.name = 'id'

df.to_csv('results-jwkim-v3.csv',index=True, header=True)