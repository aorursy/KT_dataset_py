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

from sklearn.model_selection import GridSearchCV



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
def labeling(label,n):

    arr=['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

    Y=np.zeros_like(label)

    for i in range(10):

        mask=(label==arr[i])

        Y[mask]=i



    return Y
# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_train.csv')

# 950 rows x 30 columns



feature=['tempo','beats','chroma_stft','rmse','spectral_centroid','spectral_bandwidth','rolloff','zero_crossing_rate','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20']



for i in feature:

    df_data[i] = scale(df_data[i])



sample=950

x=[]

for i in range(sample):

    x.append(np.array(df_data.ix[i][1:29]))

X=np.array(x)



label=np.array(df_data['label'][0:sample])

Y=labeling(label,sample)

Y=Y.astype('int') 



df_data
X_train,X_val,y_train,y_val=train_test_split(X,Y,test_size=0.25,random_state=42)
knn = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1, 10)}

grid = GridSearchCV(knn, param_grid, cv=5)

grid.fit(X_train,y_train)

model = grid.best_estimator_

model
y_fit = model.predict(X_val)

print(classification_report(y_fit,y_val))

print(confusion_matrix(y_fit,y_val))
# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data_T = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_test.csv')



for i in feature:

    df_data_T[i] = scale(df_data_T[i])



sample_T=50

X_test=np.zeros(28)

for i in range(sample_T):

    X_test=np.vstack((X_test,np.array(df_data_T.ix[i][1:29])))

X_test=np.delete(X_test,0,axis=0)



label=np.array(df_data_T['label'][0:sample])

Y_test=labeling(label,sample)

Y_test=Y_test.astype('int') 
result=model.predict(X_test)

print(classification_report(result,Y_test))

print(confusion_matrix(result,Y_test))
# numpy 를 Pandas 이용하여 결과 파일로 저장

import pandas as pd



result=np.append(['label'],result)

df = pd.DataFrame(result,columns=[' '])

df=df.rename(index={0:"id"})

df.to_csv('results-yk-v2.csv',index=True, header=False)

df