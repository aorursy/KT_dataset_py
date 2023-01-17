

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import numpy as np

import cv2 

import os

import glob













dataset_train = "/content/input2/train/"



dataset_train_y_name = glob.glob('../input/2019-fall-pr-project/train/train/*.jpg')





dataset_train_x = np.empty((0,32,32,3), int)

dataset_train_y = np.empty((0,1), int)

print(dataset_train_x.shape)

print(dataset_train_y.shape)



for i in range (2000):

  image = cv2.imread(dataset_train_y_name[i])

  

  if os.path.basename(dataset_train_y_name[i])[0:3]=='dog':

    dataset_train_y = np.vstack([dataset_train_y, 1])

  if os.path.basename(dataset_train_y_name[i])[0:3]=='cat':

    dataset_train_y = np.vstack([dataset_train_y, 0])

    

  image = cv2.resize(image, dsize=(32, 32))

  dataset_train_x = np.vstack([dataset_train_x, image[None,...]])

  

  





print(dataset_train_x.shape)

print(dataset_train_y.shape)















from sklearn.model_selection import train_test_split



dataset_train_x = np.ravel(dataset_train_x)

dataset_train_x = dataset_train_x.reshape(2000,3072)





Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataset_train_x, dataset_train_y, random_state=42)

















from sklearn.svm import SVC

from sklearn.decomposition import PCA as RandomizedPCA



from sklearn.pipeline import make_pipeline





print(Xtrain.shape)

print(Ytrain.shape)

print(Xtest.shape)

print(Ytest.shape)





#설정

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)

svc = SVC(kernel='rbf', class_weight='balanced')

model = make_pipeline(pca, svc)



from sklearn.model_selection import GridSearchCV



param_grid = {'svc__C': [5, 6, 7, 8],

              'svc__gamma': [0.005, 0.007 ,0.01]}

grid = GridSearchCV(model, param_grid)



grid.fit(Xtrain, Ytrain)

print('파라미터 결정 : ', grid.best_params_)



model = grid.best_estimator_  #최고로 좋은 모델

yfit = model.predict(Xtest)   #로 예측



from sklearn.metrics import classification_report

print('정확도 : \n', classification_report(Ytest, yfit))











dataset_test = "/content/input2/test1/"



dataset_test_y_name = glob.glob('../input/2019-fall-pr-project/test1/test1/*.jpg')





dataset_test_x = np.empty((0,32,32,3), int)



print(dataset_test_x.shape)



for i in range (5000):

  image = cv2.imread(dataset_test_y_name[i])

    

  image = cv2.resize(image, dsize=(32, 32))

  dataset_test_x = np.vstack([dataset_test_x, image[None,...]])

  

  





print(dataset_test_x.shape)





dataset_test_x = np.ravel(dataset_test_x)

dataset_test_x = dataset_test_x.reshape(5000,3072)







print(dataset_test_x.shape)



















testpre = model.predict(dataset_test_x)   #로 예측









# numpy 를 Pandas 이용하여 결과 파일로 저장





import pandas as pd



result = { 'label' : testpre }

df = pd.DataFrame(testpre)

df = df.replace('dog',1)

df = df.replace('cat',0)







df.to_csv('results-baeksohyeon.csv',index=True, header=False)
! kaggle competitions submit -c 2019-fall-pr-project -f results-baeksohyeon.csv -m "baeksohyeon"