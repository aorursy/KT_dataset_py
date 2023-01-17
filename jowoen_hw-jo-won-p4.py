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


from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split



import numpy as np



import cv2 

import os
import matplotlib.pyplot as plt

import os

from tqdm import tqdm

import scipy.misc



dataset_train = "/kaggle/input/2019-fall-pr-project/train/train/"

filename = os.listdir(dataset_train)



filename.sort()



#k = plt.imread(dataset_train)

X = np.zeros([len(filename),32*32*3])





for i, name in tqdm(enumerate(filename), desc='Loading..'):

    

    

    image = plt.imread(dataset_train + name)

    

    image = cv2.resize(image,dsize=(32,32), interpolation=cv2.INTER_LANCZOS4)

    #image = np.resize(image,[32,32,3])

    image = image.reshape(-1)

    X[i] = image
labeling = np.zeros_like(filename)



for i, name in tqdm(enumerate(filename), desc='Loading..'):

    

    if name.split('.')[0] == 'cat':

        labeling[i] = 0

    elif name.split('.')[0] == 'dog':

        labeling[i] = 1

    else:

        break
labeling[:]

X = cv2.GaussianBlur(X,(5,5),0)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(X)

X_norm = scaler.transform(X)
X_norm.shape

from sklearn.svm import SVC

from sklearn.decomposition import PCA as RandomizedPCA

from sklearn.metrics import classification_report

import time

from sklearn.model_selection import GridSearchCV

from sklearn import metrics 

from sklearn.pipeline import make_pipeline



pca = RandomizedPCA(n_components=32, whiten=True, random_state=42)

svc = SVC(kernel='rbf', class_weight='balanced')

model = make_pipeline(pca, svc)





from sklearn.model_selection import GridSearchCV



param_grid = {'svc__C':[1,5,10,50],

             'svc__gamma':[1E-4,5E-4,1E-3,5E-3]}

grid = GridSearchCV(model, param_grid)



%time grid.fit(X_norm, labeling)

print(grid.best_params_)

model = grid.best_estimator_
ans_rbf=model.predict(X_norm)

#print("training Runtime: %0.20f Seconds"%(time.time()-start))



print(classification_report(labeling, ans_rbf))







# 분류 결과 시각화를 원하면 주석을 삭제 할 것

print("Confusion matrix:\n%s" % metrics.confusion_matrix(labeling, ans_rbf))
dataset_test = "/kaggle/input/2019-fall-pr-project/test1/test1/"





filename_test = os.listdir(dataset_test)







#k = plt.imread(dataset_train)

X_test = np.zeros([len(filename_test),32*32*3])





for i, name_t in tqdm(enumerate(filename_test), desc='Loading..'):

    

    image = plt.imread(dataset_test + name_t)

    

    image = cv2.resize(image,dsize=(32,32), interpolation=cv2.INTER_LANCZOS4)

    #image = np.resize(image,[32,32,3])

    image = image.reshape(-1)

    X_test[i] = image
X_test = cv2.GaussianBlur(X_test,(5,5),0)



from sklearn.preprocessing import StandardScaler





scaler = StandardScaler()

scaler.fit(X_test)

X_norm_t = scaler.transform(X_test)
X_norm_t


result = model.predict(X_norm_t)



result.reshape(-1,1)
import pandas as pd





result = result.reshape(-1,1)

print(result.shape)

df = pd.DataFrame(result, columns=["label"])

df.index = np.arange(1,len(df)+1)

df.index.name = 'id'



df.to_csv('results-jwon-v2.csv',index=True, header=True)

!kaggle competitions submit -c 2019-fall-pr-project -f results-jwon-v2.csv -m "jwon-181029"