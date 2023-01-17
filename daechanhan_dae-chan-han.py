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


# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_train.csv')

imgs=[]

labels=[]

#import pdb;pdb.set_trace()

features=[]

count=0

y_train=[]

for cnt in range(len(df_data['filename'])):

  feature=[]

  for i in df_data.keys():

    if i=='filename':

      #import pdb;pdb.set_trace()

      labels= df_data[i][cnt].split('.')[0]

      if labels=='blues':

        label=0

      if labels=='country':

        label=2

      if labels=='rock':

        label=9

      if labels=='jazz':

        label=5

      if labels=='reggae':

        label=8

      if labels=='hiphop':

        label=4

      if labels=='classical':

        label=1

      if labels=='disco':

        label=3

      if labels=='pop':

        label=7

      if labels=='metal':

        label=6

        

      y_train.append(label)

      continue

    elif i=='label':

      break

    feature.append(df_data[i][cnt])

  #import pdb;pdb.set_trace()

  features.append(feature)

 



  





features= scale(features)

features=np.array(features)

features.shape

y_train=np.array(y_train)





features.shape

y_train.shape




from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.metrics import classification_report

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],

              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),

                   param_grid, cv=5, iid=False,verbose=1)

SVM=clf.fit(features, y_train)



#clf = clf.fit(X_train, y_train)

label=SVM.predict(features)



print(classification_report(y_train,label))

















# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_test.csv')



imgstest=[]

labelstest=[]

#import pdb;pdb.set_trace()

featurestest=[]

count=0

y_test=[]

ids=[]

for cnt in range(len(df_data['filename'])):

  feature=[]

  labels= int(df_data['filename'][cnt].split('.')[0])



  for i in df_data.keys():

    if i=='filename':

      continue

    elif i=='label':

      continue

    feature.append(df_data[i][cnt])

  #import pdb;pdb.set_trace()

  ids.append(labels)

  featurestest.append(feature)

 













featurestest= scale(featurestest)

featurestest=np.array(featurestest)

featurestest.shape

#y_test=np.array(y_test)



result=SVM.predict(featurestest)



import pandas as pd

ids=[]

for i in range(1,result.shape[0]+1):

  ids.append(i)



df=pd.DataFrame({'id': ids,'label': result})



df.to_csv('results-dc-v2.csv',index=False, header=True)



! kaggle competitions submit -c 2019-pr-midterm-musicclassification -f results-dc-v2.csv -m "dc-111108"


