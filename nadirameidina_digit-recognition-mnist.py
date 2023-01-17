# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
sampesepuluh = [1,2,4,7,9,11,12,14,18,42]
i = 0
while(i <= 42):
    i+=1
    if i in sampesepuluh:
        img=train_images.iloc[i].values
        img=img.reshape((28,28))
        plt.imshow(img,cmap='gray')
        plt.title(train_labels.iloc[i,0])
        plt.show()
        #print(i)
sampesepuluh = [1,2,4,7,9,11,12,14,18,42]
i = 0
while(i <= 42):
    i+=1
    if i in sampesepuluh:
        plt.hist(train_images.iloc[i])
        plt.show()
# Todo
print(train_labels.iloc[:5])
data1 = train_images.iloc[1]
data2 = train_images.iloc[3]
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.append(data1,data2)
print(len(data3))
plt.hist(data3)
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
print(train_labels.values.ravel())
print(np.unique(test_labels))
sampesepuluh = [1,2,4,7,9,11,12,14,18,42]
i = 0
test_images[test_images>0]=1
train_images[train_images>0]=1
while(i <= 42):
    i+=1
    if i in sampesepuluh:
        img=train_images.iloc[i].as_matrix().reshape((28,28))
        plt.imshow(img,cmap='binary')
        plt.title(train_labels.iloc[i])
        plt.show()
sampesepuluh = [1,2,4,7,9,11,12,14,18,42]
i = 0
while(i <= 42):
    i+=1
    if i in sampesepuluh:
        plt.hist(train_images.iloc[i])
        plt.show()
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])
print(results)
print(len(results))
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
print(os.listdir("."))
# from https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel

# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(df)
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

labeled_images = pd.read_csv('../input/train.csv')
y = labeled_images.label
pixel = ['pixel1','pixel2','pixel3']
X = labeled_images[pixel]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 47, test_size = 0.25)

clf = DecisionTreeClassifier(criterion = 'entropy')
clf.fit(X_train, y_train)
y_pred =  clf.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))