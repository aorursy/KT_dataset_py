# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import cv2
import matplotlib.pyplot as plt
import os
data=[]
labels=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if (len(str(os.path.join(dirname,filename)).split('/'))>5):
            #print(os.path.join(dirname, filename))
            img=cv2.imread(os.path.join(dirname,filename))
            img=cv2.resize(img,(64,64))
            label=str(os.path.join(dirname,filename)).split('/')[-2]
            data.append(img)
            labels.append(label)
            

# Any results you write to the current directory are saved as output.
print(len(data))
print(data[0].shape)
len(labels)
import seaborn as sns
sns.countplot(data=pd.DataFrame(labels,columns=['value']),x='value')
plt.show()
plt.imshow(data[0])
image=[]
for i in data:
    image.append((i/255.0).reshape(64*64*3))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(image,labels,random_state=101,test_size=0.25)
from sklearn.svm import SVC
model=SVC(kernel='rbf')
model.fit(X_train,y_train)
y_ptr=model.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,y_ptr))
print(confusion_matrix(y_test,y_ptr))
print(classification_report(y_test,y_ptr))
