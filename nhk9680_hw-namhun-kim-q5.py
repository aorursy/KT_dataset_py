# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from pandas.tools.plotting import scatter_matrix

#from pandas.plotting import autocorrelation_plot



#import matplotlib.pyplot as plt

#from matplotlib.colors import ListedColormap

#from mpl_toolkits.mplot3d import axes3d, Axes3D

#import seaborn as sns



from sklearn.preprocessing import scale

#from sklearn.preprocessing import StandardScaler

#from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedShuffleSplit



from sklearn.svm import SVC

#from sklearn.neighbors import NearestCentroid

#from sklearn.neighbors import KNeighborsClassifier

#from sklearn.naive_bayes import MultinomialNB

from sklearn.decomposition import PCA



#from sklearn.model_selection import train_test_split

#from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

#from sklearn.metrics import confusion_matrix

#from sklearn import metrics



#from itertools import product



import warnings

warnings.filterwarnings('ignore')





# Any results you write to the current directory are saved as output.
# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('../input/2019-pr-midterm-musicclassification/data_train.csv')

df_data.drop(['filename'], axis=1, inplace=True)
split = StratifiedShuffleSplit(test_size=0.25,  random_state=42)



for train_idx, val_idx in split.split(df_data, df_data["label"]):

    train = df_data.iloc[train_idx]

    val = df_data.iloc[val_idx]    



X_train = train.drop('label', axis=1)

Y_train = train['label']



X_val = val.drop('label', axis=1)

Y_val = val['label']



X_train = scale(X_train)

X_val = scale(X_val)
svc = SVC(kernel = 'poly')

svc.fit(X_train, Y_train)
predict = svc.predict(X_val)

print(classification_report(Y_val, predict))
param_grid = [{'kernel':["poly"], \

               'degree': [1,2,3,4,5],\

               'gamma': [0.01, 0.1, 0.5, 1], \

               'coef0': [0, 0.1, 0.5, 1], \

               'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}]

grid_search = GridSearchCV(svc, param_grid, cv=10, n_jobs=-1)

grid_search.fit(X_val, Y_val)
grid_search.best_params_
svc = SVC(C=5, kernel='poly', degree=4)
svc.fit(X_train, Y_train)
predict = svc.predict(X_val)

print(classification_report(Y_val, predict))


# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('../input/2019-pr-midterm-musicclassification/data_test.csv')

df_data.drop(['filename'], axis=1, inplace=True)



X_test = df_data.drop(['label'], axis=1)



X_test = scale(X_test)
result = svc.predict(X_test)


# numpy 를 Pandas 이용하여 결과 파일로 저장





import pandas as pd



#print(result.shape)

df = pd.DataFrame(result, columns=['label'])



'''

df = df.replace('blues', 0)

df = df.replace('country', 1)

df = df.replace('rock', 2)

df = df.replace('jazz', 3)

df = df.replace('reggae', 4)

df = df.replace('hiphop', 5)

df = df.replace('classical', 6)

df = df.replace('disco', 7)

df = df.replace('pop', 8)

df = df.replace('metal', 9)

'''

df = df.replace('blues', 0)

df = df.replace('classical', 1)

df = df.replace('country', 2)

df = df.replace('disco', 3)

df = df.replace('hiphop', 4)

df = df.replace('jazz', 5)

df = df.replace('metal', 6)

df = df.replace('pop', 7)

df = df.replace('reggae', 8)

df = df.replace('rock', 9)



df.index += 1

df.index.name = 'id'



df.to_csv('result_namhun_kim-2316.csv',index=True, header=True)