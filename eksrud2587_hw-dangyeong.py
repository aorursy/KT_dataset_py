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


import pandas as pd

# Load datasets

# DataFrame 을 이용하면 편리하다.



df_data = pd.read_csv('../input/2019-pr-midterm-musicclassification/data_train.csv')

df_data=pd.DataFrame(df_data)



datas=[]

for i in range(len(df_data)):

  x=(df_data.iloc[i:i+1,1:29])

  x_new=scale(x,axis=0,with_mean=0)

  datas.append(x_new)

  

labels=[]

data = pd.read_csv("../input/2019-pr-midterm-musicclassification/data_train.csv",encoding = "utf-8")



y = data["label"]

y = y.replace("blues",int(0))

y = y.replace("country",int(2))

y = y.replace("rock",int(9))

y = y.replace("jazz",int(5))

y = y.replace("reggae",int(8))

y = y.replace("hiphop",int(4))

y = y.replace("classical",int(1))

y = y.replace("disco",int(3))

y = y.replace("pop",int(7))

y = y.replace("metal",int(6))



df=pd.DataFrame(y)



for i in range(len(df)):

  labels.append(df.loc[i][-1])



datas=np.reshape(datas,[-1,28])
train_d, test_d, train_l, test_l = train_test_split(datas,labels, test_size = 0.75,random_state = 42)
from sklearn.svm import LinearSVC,SVC

from sklearn.svm import SVC



clf=SVC(kernel='linear',C=0.01)

#clf=SVC(kernel='poly',C=1,degree=5,coef0=1)

clf.fit(train_d,train_l)


# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('../input/2019-pr-midterm-musicclassification/data_test.csv')



result=[]

for i in range(0,50):

  x=(df_data.iloc[i:i+1,1:29])

  x_new=scale(x,axis=0,with_mean=0)

  result.append(x_new)

result=np.reshape(result,[-1,28])
yfit=clf.predict(result)


import pandas as pd



ID=[]

for i in range(1,51):

  ID.append(i)

  

id=pd.Series(ID)

result=pd.Series(yfit)



array={}

array['id']=ID

array['label']=result



df = pd.DataFrame(array)

print(df)



#print(result.shape)



df.to_csv('results-yk-v2.csv',index=False, header=True)