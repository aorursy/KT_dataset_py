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






from sklearn.preprocessing import StandardScaler





scaler = StandardScaler()



# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_train.csv')



dataset = df_data.loc[:,:].values[:,1:-1]





dataset = scaler.fit(dataset).transform(dataset)





label = df_data.values[:,-1]
print(dataset)

print(label)
labeling = np.zeros_like(label)

k = -1

for i, data in enumerate(label):

    if label[i] != label[i-1]:

        k = k+1

    labeling[i] = k
labeling.shape
np.random.seed(42)



validationInd = np.random.choice(np.arange(950), round(950/4), replace=False)



trainInd = np.setdiff1d(np.arange(950), validationInd)
from sklearn.svm import SVC

from sklearn.decomposition import PCA as RandomizedPCA

from sklearn.metrics import classification_report

import time

from sklearn.model_selection import GridSearchCV

from sklearn import metrics 

from sklearn.pipeline import make_pipeline



pca = RandomizedPCA(n_components=28, whiten=True, random_state=42)

svc = SVC(kernel='rbf', class_weight='balanced')

model = make_pipeline(pca, svc)





from sklearn.model_selection import GridSearchCV



param_grid = {'svc__C':[1,5,10,50],

             'svc__gamma':[1E-4,5E-4,1E-3,5E-3]}

grid = GridSearchCV(model, param_grid)



labeling=labeling.astype('int')
%time grid.fit(dataset[trainInd], labeling[trainInd])

print(grid.best_params_)

model = grid.best_estimator_

#train_test

from sklearn.metrics import classification_report

from sklearn import metrics 

tr_predict = model.predict(dataset[trainInd])

print(classification_report(labeling[trainInd], tr_predict))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(labeling[trainInd], tr_predict))
#validation_test

from sklearn.metrics import classification_report

from sklearn import metrics 

val_predict = model.predict(dataset[validationInd])

print(classification_report(labeling[validationInd], val_predict))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(labeling[validationInd], val_predict))




from sklearn.preprocessing import StandardScaler





scaler = StandardScaler()



df_data_test = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_test.csv')



allData_test = df_data_test.loc[:,:].values[:,1:-1]



allNormData_test = scaler.fit(allData_test).transform(allData_test)







allNormData_test.shape


#test_test

from sklearn.metrics import classification_report

from sklearn import metrics 

test_predict = model.predict(allNormData_test)



test_predict.shape
result = test_predict
import pandas as pd

ranges=np.ones(result.shape[0])

for i in range(result.shape[0]):

    ranges[i]=int(i+1)

print(result.shape)

df = pd.DataFrame({'id':ranges,'label':result})

df.to_csv('results-jwon-v2_5.csv',index=False, header=True)

! kaggle competitions submit -c 2019-pr-midterm-musicclassification -f results-jwon-v2_5.csv -m "jwon-191031_p5"