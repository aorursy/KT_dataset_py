# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#print(type(np))
#print(type(pd))
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
#print(type(plt))
#print(type(train_test_split))
#print(type(svm))
#print(type(tree))
#%matplotlib inline
# load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
#print(images)
#print(labels)
#print(train_images)
#print(test_images)
#print(train_labels)
#print(type(labeled_images.iloc))
# now we gonna load the second image, reshape it as matrix than display it
i=1
#print(train_labels)
img=train_images.iloc[i].values
#print(img)
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
#print(img)
#print(plt.imshow)
#print(plt.title)
#for i in range(len(train_labels)):
    #print(i, train_labels.iloc[i, 0])
#label 0 : indeks 12
#1 : 4
#2 : 18 
#3 : 2
#4 : 42
#5 : 9
#6 : 1
#7 : 0
#8 : 13
#9 : 62

#Membuat gambar sesuai banyak label kelas 
array = [12, 4, 18, 2, 42, 9, 1, 0, 13, 62]
for i in array:
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[i,0])
    plt.show()
    #plt.close()
    
train_images.iloc[i].describe()
#print(train_images.iloc[i])
#print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])
# create histogram for each class (data merged per class)
# Todo
#print(train_images.iloc[i])
#print(train_labels.iloc[:5])
data1 = train_images.iloc[1]
#print(train_images.iloc[1])
data2 = train_images.iloc[3]
#print(train_images.iloc[3])
#data4 = train_images.iloc[4]
data1 = np.array(data1)
#print(data1)
data2 = np.array(data2)
data3 = np.append(data1,data2)
print(len(data3))
plt.hist(data3)

# create histogram for each class (data merged per class)
# Todo

#histogram data perkelas label
array = [12, 4, 18, 2, 42, 9, 1, 0, 13, 62]
for i in array:
    img=train_images.iloc[i].values
    data = np.array(img)
    plt.hist(data)
    plt.title(train_labels.iloc[i,0])
    plt.show()

#histogram data untuk semua kelas label 
array = [12, 4, 18, 2, 42, 9, 1, 0, 13, 62]
img = train_images.iloc[0]
data = np.append(img, array)
plt.hist(data)
plt.show()
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# Put your verification code here
# Todo

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

data_model = DecisionTreeRegressor(random_state = 0)
data_model.fit(train_images, train_labels)

predictions = data_model.predict(test_images)
MAE = mean_absolute_error(predictions, test_labels)
print(MAE)

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