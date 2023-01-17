# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
print ('Setup Complete')
# load the data
labeled_images = pd.read_csv('../input/mnist-data/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
print(images)
print (type(labeled_images))
labeled_images.describe()
# now we gonna load the second image, reshape it as matrix than display it
i=1
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
# Todo: Put your code here
arr = [[12],[4],[18],[6],[42],[9],[1],[23],[11],[7]]
#print(label)
for i in range(10):
    j = arr[i][0]
    img = train_images.iloc[j].values
    img = img.reshape((28,28))
    plt.figure()
    plt.imshow(img,cmap = 'gray')
    data = 'class ' + str(train_labels.iloc[j,0])
    plt.title(data)
    plt.show()
    plt.close()
train_images.iloc[i].describe()
print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])
plt.title(i)
plt.xlabel("long of pixels")
plt.ylabel("number of columns")
# create histogram for each class (data merged per class)
# Todo
print(train_labels.iloc[:5])
data1 = train_images.iloc[1]
data2 = train_images.iloc[3]
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.append(data1,data2)
print(len(data3))
plt.hist(data3)
# Answer Q3
# create histogram for each class (data merged per class)
# Todo
for i in range(10):
    a = arr[i][0]
    data1 = train_images.iloc[a]
    data2 = np.array(data1)
    plt.figure()
    data = 'class ' + str(train_labels.iloc[a,0])
    plt.title(data)
    plt.hist(data3)
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# Put your verification code here
# Todo
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

digit_model = DecisionTreeRegressor(random_state = 0)
digit_model.fit(train_images,train_labels)
predictions = digit_model.predict(test_images)

print(train_labels.values.ravel())
print(np.unique(test_labels)) 
print(mean_absolute_error(test_labels, predictions)) 
# to see class number
test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
# now plot again the histogram
plt.hist(train_images.iloc[i])
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
from time import time
from time import sleep

#The time taken to search for results in the pixel case data after normalization
clf.fit(train_images, train_labels.values.ravel())
a = float(time())
clf.score(test_images,test_labels)
b = float(time())
elapsed = b - a
print("Waktu1 : ", elapsed)

#The time needed to find the score in the case of the pixel data before normalization
train_image, test_image,train_label, test_label = train_test_split(images, labels, test_size=0.2, random_state=0)
clf.fit(train_image, train_label.values.ravel())
a = float(time())
clf.score(test_image,test_label)
b = float(time())
elapsed = b - a
print("Waktu2 : ", elapsed)
# Test again to data test
test_data=pd.read_csv('../input/mnist-data/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])
# separate code section to view the results
print(results)
print(len(results))
# dump the results to 'results.csv'
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
#check if the file created successfully
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
from sklearn.tree import DecisionTreeClassifier

digit_model = DecisionTreeClassifier(random_state = 1)
digit_model.fit(train_images,train_labels)
predictions = digit_model.predict(test_images)

print(train_labels.values.ravel())
print(np.unique(test_labels)) 
print(mean_absolute_error(test_labels, predictions)) 

#SVM
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

digit_model = DecisionTreeClassifier(random_state = 1)
digit_model.fit(train_images,train_labels)
predictions = digit_model.predict(test_images)
Classifier = mean_absolute_error(test_labels, predictions)

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

digit_model = DecisionTreeRegressor(random_state = 1)
digit_model.fit(train_images,train_labels)
predictions = digit_model.predict(test_images)
Regressor = mean_absolute_error(test_labels, predictions)

#SVM
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
SVM = clf.score(test_images,test_labels)

print('Classifier : ', Classifier)
print('Regressor : ', Regressor)
print('SVM : ', SVM)
from sklearn.tree import DecisionTreeClassifier

digit_model = DecisionTreeClassifier(random_state = 0)
digit_model.fit(train_images,train_labels)
predictions = digit_model.predict(test_images)
 
print(mean_absolute_error(test_labels, predictions)) 