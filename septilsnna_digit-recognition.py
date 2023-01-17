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
#initialized variable a, b, c, d for Question 6 (4)
a = train_images
b = test_images
c = train_labels
d = test_labels
# now we gonna load the second image, reshape it as matrix than display it
i=1
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
#create an index list that the value number is 0, 1, 2, ..., 8, 9
num = [12, 4, 18, 2, 42, 9, 1, 0, 11, 7]
#use for loop to display the result
for i in num:
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.figure()
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[i,0])
#train_images.iloc[i].describe()
#print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])
# create histogram for each class (data merged per class)
num = [12, 4, 18, 2, 42, 9, 1, 0, 11, 7]
for i in num:
    plt.figure()
    plt.hist(train_images.iloc[i])
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number
test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.loc[train_labels.index[i],'label'])
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
clf_model = DecisionTreeClassifier(random_state=0)
clf_model.fit(train_images, train_labels)
val_predictions = clf_model.predict(test_images)
result1 = mean_absolute_error(test_labels, val_predictions)
print(result1)

#SVM
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
result2 = clf.score(test_images,test_labels)
print(result2)
#DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
#define model
images_model = DecisionTreeRegressor(random_state=0)
#fit model
images_model.fit(train_images, train_labels)
#get predicted on validation data
val_predictions = images_model.predict(test_images)
result3 = mean_absolute_error(test_labels, val_predictions)

df = pd.DataFrame([[result3], [result1], [result2]], index=['DTR', 'DTC', "SVM"], columns=['result'])
print(df)
#define model
clf_model = DecisionTreeClassifier(random_state=0)
#fit model
clf_model.fit(a, c)
#get predicted on validation data
val_predictions = clf_model.predict(b)
print(mean_absolute_error(d, val_predictions))