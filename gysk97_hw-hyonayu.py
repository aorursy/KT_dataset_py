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
from sklearn.preprocessing import scale

# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_train.csv')

df_data = pd.DataFrame(df_data)

#print(df_data.index)

name = df_data.columns

#print(len(df_data.loc[2]))

label = []

sub_data = []

data = []

scale_value=[]

for i in range(1, 29):

  a = list(df_data[name[i]])

  #a = np.array(a)

  scale_value.append(scale(a, with_mean= 0))

  #scale_value.append(StandardScaler().fit(a))

#print(len(scale_value[0]))  



#print(list(df_data.loc[1]))

for i in range(len(df_data)):

  sub_data = []

  label.append(df_data.loc[i][-1])

  for j in range(0,len(scale_value)):

    sub_data.append(scale_value[j][i])

  data.append(sub_data)



print(data)
train_D, test_D, train_L, test_L = train_test_split(data, label, test_size = 0.75,random_state = 42)

from sklearn.metrics import accuracy_score

from sklearn import svm





print(len(train_D[0]))

clf = svm.SVC(kernel = 'rbf', random_state = 42)





clf.fit(train_D, train_L)



y = clf.predict(test_D)

accuracy_score(test_L, y)













df_data = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_test.csv')

df_data = pd.DataFrame(df_data)

#print(df_data.index)

name = df_data.columns

#print(len(df_data.loc[2]))

test_label = []

sub_data = []

test_data = []

scale_value=[]

for i in range(1, 29):

  a = list(df_data[name[i]])

  scale_value.append(scale(a, with_mean=0))

print(len(df_data))

#print(len(scale_value[0]))  



for i in range(len(df_data)):

  sub_data = []

  test_label.append(df_data.loc[i][-1])

  for j in range(0,len(scale_value)):

    sub_data.append(scale_value[j][i])

  test_data.append(sub_data)



#print(data[1])
y_predict = clf.predict(test_data)



# numpy 를 Pandas 이용하여 결과 파일로 저장





import pandas as pd

result = y_predict

name = []

data1 = {}

for i in range(1,51):

  name.append(i)

data1['id'] = name

data1['label']=result





print(result.shape)

df = pd.DataFrame(data1)

df = df.replace('blues',0)

df = df.replace('country',2)

df = df.replace('rock',9)

df = df.replace('jazz',5)

df = df.replace('reggae',8)

df = df.replace('hiphop',4)

df = df.replace('classical',1)

df = df.replace('disco',3)

df = df.replace('pop',7)

df = df.replace('metal',6)







df.to_csv('results-yk-v2.csv',index=False, header=True)