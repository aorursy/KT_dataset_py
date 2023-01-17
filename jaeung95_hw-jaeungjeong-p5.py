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

from sklearn.svm import LinearSVC, SVC

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

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')





# Any results you write to the current directory are saved as output.


df_data = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_train.csv')



df_data.drop(["filename"], axis=1, inplace=True)



label=df_data["label"].copy() # label

df_data.drop(['label'],axis=1,inplace=True)











split = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state = 42)



for train_index, cross_index in split.split(df_data, label):

    strat_train_set=(df_data.iloc[train_index])

    strat_cross_set = df_data.iloc[cross_index]

    music_labels=label[train_index]





music = strat_train_set.copy()

music = scale(music);





















clf = SVC(kernel='poly',random_state=42,class_weight='balanced')

clf.fit(music, music_labels)















# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_test.csv')

df_data.drop(["filename"], axis=1, inplace=True)







X_test = df_data.drop("label", axis=1)

y_test = df_data["label"].copy()

X_test_prepared = scale(X_test)













result=clf.predict(X_test_prepared)

accuracy_score(result,y_test)





# numpy 를 Pandas 이용하여 결과 파일로 저장

#각 장르에 따라 숫자로 변경 후 , id, label 라벨을 추가해서 저장하기



import pandas as pd

s = pd.Series(result)

data={'id':range(1,51),'label':s}

print(result.shape)

df = pd.DataFrame(data)

df.index += 1 

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

df.to_csv('results-yk-v2.csv',index=True,index_label='id', header=True,columns=["label"])