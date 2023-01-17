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
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
train_images_ori, test_images_ori ,train_labels_ori ,test_labels_ori = train_images, test_images,train_labels, test_labels
i=6
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
lst = [0]*10
for i in range (len(train_images)):
    m= train_labels.iloc[i].label
    if (0<=m<10):
        if (lst[m] == 0):
            lst[m] = i
    if 0 not in lst:
        break
print (lst)
for i in lst:
    plt.figure(i)
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
for i in lst:
    plt.figure(i)
    plt.hist(train_images.iloc[i])
print (len(train_images.iloc[1]))
c0,c1,c2,c3,c4,c5,c6,c7,c8,c9 = 0,0,0,0,0,0,0,0,0,0
arr0,arr1,arr2,arr3,arr4,arr5,arr6,arr7,arr8,arr9=[],[],[],[],[],[],[],[],[],[]
for i in range(len(train_images)):
    if (train_labels.iloc[i].label == 0):
        c0+=1
        data = train_images.iloc[i]
        data = np.array(data)
        arr0 = np.append(arr0,data)
    elif (train_labels.iloc[i].label == 1):
        c1+=1
        data = train_images.iloc[i]
        data = np.array(data)
        arr1 = np.append(arr1,data)
    elif (train_labels.iloc[i].label == 2):
        c2+=1
        data = train_images.iloc[i]
        data = np.array(data)
        arr2 = np.append(arr2,data)
    elif (train_labels.iloc[i].label == 3):
        c3+=1
        data = train_images.iloc[i]
        data = np.array(data)
        arr3 = np.append(arr3,data)
    elif (train_labels.iloc[i].label == 4):
        c4+=1
        data = train_images.iloc[i]
        data = np.array(data)
        arr4 = np.append(arr4,data)
    elif (train_labels.iloc[i].label == 5):
        c5+=1
        data = train_images.iloc[i]
        data = np.array(data)
        arr5 = np.append(arr5,data)
    elif (train_labels.iloc[i].label == 6):
        c6+=1
        data = train_images.iloc[i]
        data = np.array(data)
        arr6 = np.append(arr6,data)
    elif (train_labels.iloc[i].label == 7):
        c7+=1
        data = train_images.iloc[i]
        data = np.array(data)
        arr7 = np.append(arr7,data)
    elif (train_labels.iloc[i].label == 8):
        c8+=8
        data = train_images.iloc[i]
        data = np.array(data)
        arr8 = np.append(arr8,data)
    elif (train_labels.iloc[i].label == 9):
        c9+=1
        data = train_images.iloc[i]
        data = np.array(data)
        arr9 = np.append(arr9,data)
    
print ('label\t\t--->\t\tamount number class')
print (0,'\t\t--->\t\t',c0)
print (1,'\t\t--->\t\t',c1)
print (2,'\t\t--->\t\t',c2)
print (3,'\t\t--->\t\t',c3)
print (4,'\t\t--->\t\t',c4)
print (5,'\t\t--->\t\t',c5)
print (6,'\t\t--->\t\t',c6)
print (7,'\t\t--->\t\t',c7)
print (8,'\t\t--->\t\t',c8)
print (9,'\t\t--->\t\t',c9)

plt.figure(0)
plt.hist(arr0)
plt.xlabel('label 0')

plt.figure(1)
plt.hist(arr1)
plt.xlabel('label 1')

plt.figure(2)
plt.hist(arr2)
plt.xlabel('label 2')

plt.figure(3)
plt.hist(arr3)
plt.xlabel('label 3')

plt.figure(4)
plt.hist(arr4)
plt.xlabel('label 4')

plt.figure(5)
plt.hist(arr5)
plt.xlabel('label 5')

plt.figure(6)
plt.hist(arr6)
plt.xlabel('label 6')

plt.figure(7)
plt.hist(arr7)
plt.xlabel('label 7')

plt.figure(8)
plt.hist(arr8)
plt.xlabel('label 8')

plt.figure(9)
plt.hist(arr9)
plt.xlabel('label 9')



clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

label_model = DecisionTreeRegressor(random_state=0)
label_model.fit(train_images, train_labels.values)
prediction = label_model.predict(test_images)
mae = mean_absolute_error(prediction, test_labels)
print (mae)

test_images[test_images>0]=1
train_images[train_images>0]=1

plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])


print (lst)
for i in lst:
    test_images[test_images>0]=1
    train_images[train_images>0]=1
    
    plt.figure(i)
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.imshow(img,cmap='binary')
    plt.title(train_labels.iloc[i,0])
for i in lst:
    test_images[test_images>0]=1
    train_images[train_images>0]=1
    
    plt.figure(i)
    plt.hist(train_images.iloc[i].values)
    
# Test again to data test
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])
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
test_images[test_images>0]=1
train_images[train_images>0]=1

test = [0.001,0.01,0.1,1,5,10,50,100,500,1000,5000,10000]
listC = []
print ('c\t\tTestScore\t\tTrainScore')
for i in test:
    clf = svm.SVC(C=i)
    clf.fit(train_images, train_labels.values.ravel())
    score = clf.score(test_images,test_labels)
    clf.fit(test_images, test_labels.values.ravel())
    scoreTrain = clf.score(train_images,train_labels)
    print(i,'\t\t',score,'\t\t',scoreTrain)
    listC.append(score)

 
if all(x == listC[0] for x in lst):
    print ('c bukan merupakan faktor')
else:
    print ('c merupakan faktor')
test_images[test_images>0]=1
train_images[train_images>0]=1

vname = 'tol'
test = [0.001,0.01,0.1,1,5,10,50,100,500,1000,5000,10000]
lst = []
print (vname+'\t\tscoreTest\t\tscoreTrain')

for i in test:
    clf = svm.SVC(tol=i)
    clf.fit(train_images, train_labels.values.ravel())
    score = clf.score(test_images,test_labels)
    clf.fit(test_images, test_labels.values.ravel())
    scoreTrain = clf.score(train_images,train_labels)
    print(i,'\t\t',score,'\t\t',scoreTrain)
    lst.append(score)
 
if all(x == lst[0] for x in lst):
    print (vname+' bukan merupakan faktor')
else:
    print ( vname+ ' merupakan faktor')
test_images[test_images>0]=1
train_images[train_images>0]=1

vname = 'degree'
test = [0.001,0.01,0.1,1,5,10,50,100,500,1000,5000,10000]
lst = []
print (vname+'\t\tscoreTest\t\tscoreTrain')

for i in test:
    clf = svm.SVC(degree=i)
    clf.fit(train_images, train_labels.values.ravel())
    score = clf.score(test_images,test_labels)
    clf.fit(test_images, test_labels.values.ravel())
    scoreTrain = clf.score(train_images,train_labels)
    print(i,'\t\t',score,'\t\t',scoreTrain)
    lst.append(score)
 
if all(x == lst[0] for x in lst):
    print (vname+' bukan merupakan faktor')
else:
    print ( vname+ ' merupakan faktor')
test_images[test_images>0]=1
train_images[train_images>0]=1

vname = 'cache_size'
test = [0.001,0.01,0.1,1,5,10,50,100,500,1000,5000,10000]
lst = []
print (vname+'\t\tscoreTest\t\tscoreTrain')

for i in test:
    clf = svm.SVC(cache_size=i)
    clf.fit(train_images, train_labels.values.ravel())
    score = clf.score(test_images,test_labels)
    clf.fit(test_images, test_labels.values.ravel())
    scoreTrain = clf.score(train_images,train_labels)
    print(i,'\t\t',score,'\t\t',scoreTrain)
    lst.append(score)
 
if all(x == lst[0] for x in lst):
    print (vname+' bukan merupakan faktor')
else:
    print ( vname+ ' merupakan faktor')
test_images[test_images>0]=1
train_images[train_images>0]=1

vname = 'coef0'
test = [0.001,0.01,0.1,1,5,10,50,100,500,1000,5000,10000]
lst = []
print (vname+'\t\tscoreTest\t\tscoreTrain')

for i in test:
    clf = svm.SVC(coef0=i)
    clf.fit(train_images, train_labels.values.ravel())
    score = clf.score(test_images,test_labels)
    clf.fit(test_images, test_labels.values.ravel())
    scoreTrain = clf.score(train_images,train_labels)
    print(i,'\t\t',score,'\t\t',scoreTrain)
    lst.append(score)
 
if all(x == lst[0] for x in lst):
    print (vname+' bukan merupakan faktor')
else:
    print ( vname+ ' merupakan faktor')
test_images[test_images>0]=1
train_images[train_images>0]=1

from sklearn.tree import DecisionTreeClassifier
MLNTrain = []
MLNTest = []
test2 = [5,10,50,100,500,1000,5000,10000]

print ('max_leaf_node\tscoreTest\t\tscoreTrain')
for i in test2:
    model = DecisionTreeClassifier(max_leaf_nodes = i)
    model.fit(train_images, train_labels.values.ravel())
    score = model.score(test_images,test_labels)
    MLNTest.append(score)
    model.fit(test_images,test_labels.values.ravel())
    scoreTrain = model.score(train_images,train_labels)
    MLNTrain.append(scoreTrain)
    print (i,'\t\t',score,'\t\t',scoreTrain)
    
plt.figure(i)
plt.plot(test2,MLNTest)
plt.plot(test2,MLNTrain)
plt.title("Decision Tree Classifier")
plt.xlabel("max_leaf_nodes")
plt.ylabel("Score")
test_images[test_images>0]=1
train_images[train_images>0]=1

from sklearn.tree import DecisionTreeRegressor
MLNTrain = []
MLNTest = []
test2 = [5,10,50,100,500,1000,5000,10000]

print ('max_leaf_node\tscoreTest\t\tscoreTrain')
for i in test2:
    model = DecisionTreeRegressor(max_leaf_nodes = i)
    model.fit(train_images, train_labels.values.ravel())
    score = model.score(test_images,test_labels)
    MLNTest.append(score)
    model.fit(test_images,test_labels.values.ravel())
    scoreTrain = model.score(train_images,train_labels)
    MLNTrain.append(scoreTrain)
    print (i,'\t\t',score,'\t\t',scoreTrain)
    
plt.figure(i)
plt.plot(test2,MLNTest)
plt.plot(test2,MLNTrain)
plt.title("Decision Tree Regressor")
plt.xlabel("max_leaf_nodes")
plt.ylabel("Score")
    
    
from sklearn.tree import DecisionTreeClassifier
MLNTrain = []
MLNTest = []
test2 = [5,10,50,100,500,1000,5000,10000]

print ('max_leaf_node\tscoreTest\t\tscoreTrain')
for i in test2:
    model = DecisionTreeClassifier(max_leaf_nodes = i)
    model.fit(train_images_ori, train_labels_ori.values.ravel())
    score = model.score(test_images_ori,test_labels_ori)
    MLNTest.append(score)
    model.fit(test_images_ori,test_labels_ori.values.ravel())
    scoreTrain = model.score(train_images_ori,train_labels_ori)
    MLNTrain.append(scoreTrain)
    print (i,'\t\t',score,'\t\t',scoreTrain)
    
plt.figure(i)
plt.plot(test2,MLNTest)
plt.plot(test2,MLNTrain)
plt.title("Decision Tree Classifier")
plt.xlabel("max_leaf_nodes")
plt.ylabel("Score")