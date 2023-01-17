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



import warnings

warnings.filterwarnings('ignore')







# Any results you write to the current directory are saved as output.


# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('../input/2019-pr-midterm-musicclassification/data_train.csv')



data_x=df_data.iloc[:,1:28]

data_y=df_data.iloc[:, 29]











from sklearn.model_selection import train_test_split







Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_x, data_y, random_state=42)

























from sklearn.svm import SVC

from sklearn.decomposition import PCA as RandomizedPCA



from sklearn.pipeline import make_pipeline





print(Xtrain.shape)

print(Ytrain.shape)

print(Xtest.shape)

print(Ytest.shape)





#설정

pca = RandomizedPCA(n_components=27, whiten=True, random_state=42)

svc = SVC(kernel='rbf', class_weight='balanced')

model = make_pipeline(pca, svc)



from sklearn.model_selection import GridSearchCV



param_grid = {'svc__C': [1,5, 6, 7, 8,9,10,11],

              'svc__gamma': [0.01,0.015,0.03]}

grid = GridSearchCV(model, param_grid)



grid.fit(Xtrain, Ytrain)

print('파라미터 결정 : ', grid.best_params_)



model = grid.best_estimator_  #최고로 좋은 모델

yfit = model.predict(Xtest)   #로 예측



from sklearn.metrics import classification_report

print('정확도 : \n', classification_report(Ytest, yfit))









# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('../input/2019-pr-midterm-musicclassification/data_test.csv')





test_x=df_data.iloc[:,1:28]

test_y=df_data.iloc[:, 29]















#blues (0),classical (1),country (2),disco(3),hiphop(4),jazz (5),metal (6),pop (7),reggae(8),rock(9)



testpre = model.predict(test_x)   #로 예측

print('정확도 : \n', classification_report(test_y, testpre))

for i in range (50):

  if testpre[i]=='blues':

    testpre[i] = 0

  if testpre[i]=='country':

    testpre[i] = 2

  if testpre[i]=='rock':

    testpre[i] = 9

  if testpre[i]=='jazz':

    testpre[i] = 5

  if testpre[i]=='reggae':

    testpre[i] = 8

  if testpre[i]=='hiphop':

    testpre[i] = 4

  if testpre[i]=='classical':

    testpre[i] = 1

  if testpre[i]=='disco':

    testpre[i] = 3

  if testpre[i]=='pop':

    testpre[i] = 7

  if testpre[i]=='metal':

    testpre[i] = 6







# numpy 를 Pandas 이용하여 결과 파일로 저장





import pandas as pd



result = { "label" : testpre }

df = pd.DataFrame(testpre)

df = df.replace('dog',1)

df = df.replace('cat',0)







df.to_csv('results-baeksohyeonP5.csv',index=True, header=False)
! kaggle competitions submit -c 2019-pr-midterm-musicclassification -f results-baeksohyeonP5.csv -m "baeksohyeonP5"