#라이브러리 로드

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



#from imutils import paths

import pandas as pd

import numpy as np

#import imutils 

import cv2 

import os



from PIL import Image



import matplotlib.pyplot as plt

from sklearn.svm import SVC



import warnings

warnings.filterwarnings('ignore')
def resize32(dir):

    img= cv2.imread(dir)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray.resize(32*32)

    return img_gray
dataset_train = "../input/2019-fall-pr-project/train/train/"



Y=[]

x=[]

data=os.listdir(dataset_train)

fig,ax=plt.subplots(3,3)

for i,axi in enumerate(ax.flat):

    img= cv2.imread(dataset_train+data[i])

    axi.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    axi.set(xticks=[], yticks=[],xlabel=data[i].split('.')[0])



for i in range(20001):

    Y.append(0 if data[i].split('.')[0]=='cat' else 1)

    image=resize32(dataset_train+data[i])

    x.append(image)



X=np.array(x)
X_train,X_val,y_train,y_val=train_test_split(X,Y,test_size=0.25,random_state=42)
from sklearn.model_selection import GridSearchCV



knn = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1, 10)}

grid = GridSearchCV(knn, param_grid, cv=5)

grid.fit(X_train,y_train)

model=grid.best_estimator_

model
yfit=model.predict(X_val)

print(classification_report(yfit,y_val))

print(confusion_matrix(yfit,y_val))
dataset_test = "../input/2019-fall-pr-project/test1/test1/"



x=[]

files=os.listdir(dataset_test)

for i in range(5000):

    image=resize32(dataset_test+files[i])

    x.append(image)



test_x=np.array(x)
# 라벨: 개(1), 고양이(0)

result=model.predict(test_x)
import pandas as pd

result=np.append(['label'],result)

df = pd.DataFrame(result,columns=[' '])

df=df.rename(index={0:"id"})



df.to_csv('results.csv',index=True, header=False)

df