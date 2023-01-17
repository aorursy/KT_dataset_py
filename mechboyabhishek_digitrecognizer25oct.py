# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df_train=pd.read_csv('../input/train.csv',encoding='UTF-8')
df_test=pd.read_csv('../input/test.csv',encoding='UTF-8')
# Any results you write to the current directory are saved as output.
df_train.head(5)
df_sample=df_train.sample(n=50,random_state=100)
result=df_sample.iloc[2:3,1:]
#result
plt.imshow(np.array(result).reshape(28,28),cmap=cm.gray)
import sklearn.svm as svm
x_test=df_test.iloc[:,0:]
#y_test=df_test.iloc[:,0:0]
x_train=df_train.iloc[:,1:]
y_train=df_train.iloc[:,0:1]
y_train=np.ravel(y_train)

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

#svm_model=svm.SVC(C=1.0,degree=4, gamma='auto', kernel='rbf')
svm_model=svm.SVC(C=1,degree=8, gamma='auto', kernel='rbf')
    
svm_model.fit(X=x_train,y=y_train)
y_pred=svm_model.predict(X=x_test)
print(y_pred)
count_index=list(range(1,len(y_pred)+1,1))
submission=pd.DataFrame(columns=['ImageId','Label'])
submission['ImageId']=count_index
submission['Label']=pd.DataFrame(data=y_pred)
submission.to_csv('submission8.csv', encoding='utf-8',index=False)
#print(os.listdir("../working"))