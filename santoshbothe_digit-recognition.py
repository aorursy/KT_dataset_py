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
#%matplotlib inline
# load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
# now we gonna load the second image, reshape it as matrix than display it
i=1
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
# Todo: Put your code here
arr = [[],[],[],[],[],[],[],[],[],[]]
for i in range(len(train_labels)):
    a = train_labels.iloc[i, 0]
    arr[a] += [i]
    
for i in range(10):
    j = arr[i][0]
    img = train_images.iloc[j].values
    img = img.reshape((28,28))
    fig = plt.figure()
    plt.imshow(img,cmap = 'gray')
    judul = 'Kelas ' + str(train_labels.iloc[j,0])
    plt.title(judul)
    plt.show()
    plt.close()

train_images.iloc[i].describe()
print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])
# create histogram for each class (data merged per class)
# Todo
#print(train_labels.iloc[:5])
data1 = train_images.iloc[1]
data2 = train_images.iloc[3]
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.append(data1,data2)
print(len(data3))
plt.hist(data3)
# create histogram for each class (data merged per class)
# Todo
# This is the progam to answer Q3
for i in range(10):
    j = arr[i][0]
    data = train_images.iloc[j]
    data = np.array(data)
    plt.figure()
    plt.hist(data)
    plt.title('Histogram Kelas ' + str(train_labels.iloc[j,0]))
    plt.show()
    plt.close()

semua_data = np.append(train_images.iloc[data[0]], (train_images.iloc[data[1:]]))
print(len(semua_data))
plt.figure()
plt.hist(semua_data)
plt.title("Histogram Setiap Kelas")
plt.show()
plt.close()
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# Put your verification code here
# Todo
print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number
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
# Test again to data test
test_data=pd.read_csv('../input/test.csv')
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