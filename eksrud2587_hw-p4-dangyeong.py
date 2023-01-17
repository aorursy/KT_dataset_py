

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split



import numpy as np



import cv2 

import os



dataset_train = "../input/2019-fall-pr-project/train/train/"



datas = []

label = []

cat_data = []

dog_data = []



#cat.255.jpg



for i in os.listdir(dataset_train):

  

  if 'cat' == i.split('.')[0]:

    data=cv2.imread(dataset_train + i)

    data= cv2.resize(data, (32,32))

    cat_data.append(data.ravel())

    label.append('0')

    if len(cat_data) == 5000 : break



  else:

    data = cv2.imread(dataset_train + i)

    data= cv2.resize(data, (32,32))

    dog_data.append(data.ravel())

    label.append('1')

    if len(dog_data) == 5000 : break

      

for c in cat_data:

  datas.append(c)

for d in dog_data:

  datas.append(d)



#print(datas)
train_d, test_d, train_l, test_l = train_test_split(datas, label, test_size = 0.75,random_state = 42)



from sklearn.svm import LinearSVC,SVC

from sklearn.svm import SVC





model=SVC(kernel='linear',C=1E10)

model.fit(train_d,train_l)

 


dataset_test = "../input/2019-fall-pr-project/test1/test1/"



result=[]

for i in os.listdir(dataset_test):

  data=cv2.imread(dataset_test+i)

  data=cv2.resize(data,(32,32))

  result.append(data.ravel())



#print(result)
yfit=model.predict(result)



# numpy 를 Pandas 이용하여 결과 파일로 저장





import pandas as pd



ID=[]

for i in range(1,5001):

  ID.append(i)

  

id=pd.Series(ID)

result=pd.Series(yfit)



array={}

array['id']=ID

array['label']=result



df = pd.DataFrame(array)

print(df)



print(result.shape)

df = df.replace('dog',1)

df = df.replace('cat',0)



df.to_csv('results-yk-v2.csv',index=False, header=True)