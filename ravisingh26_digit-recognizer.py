# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('../input/digit-recognizer/train.csv',engine='c',encoding='utf-8')
test=pd.read_csv('../input/digit-recognizer/test.csv',engine='c',encoding='utf-8')
train.info()
#images are of (1,784) shape, can be converted into(28,28)
train.head() 
print(train.shape)
print(test.shape)
#We need to convert first 5 rows to numpy arrays, so that we can use plt.imshow function
images=np.array(train.iloc[:5,1:])
labels=np.array(train.iloc[:5,:1])

plt.figure()
for index,(image,label) in enumerate(zip(images,labels)):
    #print(index,(image,label))
    plt.subplot(1,5,index+1)
    newim=image.reshape((28,28))
    plt.imshow(newim) #you can choose your own cmap as well
    plt.title('D:%d'%(label))
   


print(train['label'].value_counts())

x_train=train.drop('label',axis=1)
y_train=train['label']

model=LogisticRegression(multi_class='multinomial')
model.fit(x_train,y_train)
prob=model.predict_proba(x_train)
print(prob)
y_pred=model.predict(x_train)
comparison=pd.DataFrame({'Original':y_train,'Predicted':y_pred})
print(comparison)

cm=confusion_matrix(y_train, y_pred)
print(cm)
ac=accuracy_score(y_train, y_pred)
print(ac)
sns.heatmap(cm,annot=True,cmap='Blues_r')
plt.xlabel('Predicted')
plt.ylabel('Originial')
title='Accuracy:%f'%(ac)
plt.title(title)

#on test data
pred=model.predict(test)
submission=pd.concat([pd.Series(range(1,28001),name='ImageId'),pd.DataFrame(pred,columns=['label'])],axis=1)

#submission.to_csv('D:/python/Submission.csv')