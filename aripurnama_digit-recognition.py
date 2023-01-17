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
    plt.figure()
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
#data1 = train_images.iloc[1]
#data2 = train_images.iloc[3]
#data1 = np.array(data1)
#data2 = np.array(data2)
#data3 = np.append(data1,data2)
#print(len(data3))
#plt.hist(data3)
# create histogram for each class (data merged per class)
# Todo
# This is the progam to answer Q3

# Menampilkan histogram untuk perkelas
for i in range(10):
    j = arr[i][0]
    data = train_images.iloc[j]
    data = np.array(data)
    plt.figure()
    plt.hist(data)
    plt.title('Histogram Kelas ' + str(train_labels.iloc[j,0]))
    plt.xlabel('Nilai Pixel')
    plt.ylabel('Jumlah Kolom')
    plt.show()
    plt.close()
# Menampilkan histogram untuk kelas 0 sampai 9   
semua_data = np.append(train_images.iloc[data[0]], (train_images.iloc[data[1:]]))
plt.figure()
plt.hist(semua_data)
plt.title("Histogram Kelas 0 sampai 9")
plt.xlabel('Nilai Pixel')
plt.ylabel('Jumlah Kolom')
plt.show()
plt.close()
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# Put your verification code here
# Todo
# print(train_labels.values.ravel())
# print(np.unique(test_labels)) # to see class number

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

clf_split_model = DecisionTreeRegressor()
clf_split_model.fit(train_images, train_labels)
test_predictions = clf_split_model.predict(test_images)
print(mean_absolute_error(test_labels, test_predictions))
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

# Waktu yang dibutuhkan untuk mencari skor pada kasus data pixel setelah dinormalisasi
clf.fit(train_images, train_labels.values.ravel())
a = float(time())
clf.score(test_images,test_labels)
b = float(time())
elapsed = b - a
print("Waktu yang dibutuhkan: ", elapsed)

# Waktu yang dibutuhkan untuk mencari skor pada kasus data pixel sebelum dinormalisasi
train_img, test_img,train_lbl, test_lbl = train_test_split(images, labels, test_size=0.2, random_state=0)
clf.fit(train_img, train_lbl.values.ravel())
a = float(time())
clf.score(test_img,test_lbl)
b = float(time())
elapsed = b - a
print("Waktu yang dibutuhkan: ", elapsed)

# Test again to data test
clf.fit(train_images, train_labels.values.ravel())
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

def getTrainMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels):
    modelTrain = DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    modelTrain.fit(train_images, train_labels)
    predictTrain = modelTrain.predict(train_images)
    mae = mean_absolute_error(train_labels, predictTrain)
    return (mae)

def getTestMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels):
    modelTest = DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    modelTest.fit(train_images, train_labels)
    predictTest = modelTest.predict(test_images)
    mae = mean_absolute_error(test_labels, predictTest)
    return (mae)

trainMAE = []
testMAE = []

bestTrain = {}
bestTest = {}

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
print("Train MAE for each nodes:")
for max_leaf_nodes in candidate_max_leaf_nodes:
    myTrainMAE = getTrainMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    trainMAE.append(myTrainMAE)
    bestTrain.update({max_leaf_nodes : myTrainMAE})
    print("Max leaf nodes: %d  \t\t Train MAE:  %d" %(max_leaf_nodes, myTrainMAE))

print("\nTest MAE for each nodes:")
for max_leaf_nodes in candidate_max_leaf_nodes:
    myTestMAE = getTestMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    testMAE.append(myTestMAE)
    bestTest.update({max_leaf_nodes : myTestMAE})
    print("Max leaf nodes: %d  \t\t Test MAE:  %d" %(max_leaf_nodes, myTestMAE))

best_tree_size_train  = min(bestTrain, key=bestTrain.get)
best_tree_size_test = min(bestTest, key=bestTest.get)
print("\nTrain MAE for each nodes: ", trainMAE)
print("Test MAE for each nodes: ", testMAE)
print("\nBest classifier tree size for train = ", best_tree_size_train)
print("Best classifier tree size for test = ", best_tree_size_test)


plt.figure()
plt.plot(candidate_max_leaf_nodes, trainMAE, color="red")
plt.plot(candidate_max_leaf_nodes, testMAE, color="green")
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Classifier")
plt.show()

CLS = DecisionTreeClassifier(max_leaf_nodes=best_tree_size_train, random_state=0)
CLS.fit(train_images,train_labels)
scoreCLS = CLS.score(test_images,test_labels)
print("Score Decision Tree Classifier = ",scoreCLS)
def getTrainMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels):
    modelTrain = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    modelTrain.fit(train_images, train_labels)
    predictTrain = modelTrain.predict(train_images)
    mae = mean_absolute_error(train_labels, predictTrain)
    return (mae)

def getTestMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels):
    modelTest = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    modelTest.fit(train_images, train_labels)
    predictTest = modelTest.predict(test_images)
    mae = mean_absolute_error(test_labels, predictTest)
    return (mae)

trainMAE = []
testMAE = []

bestTrain = {}
bestTest = {}

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
print("Train MAE for each nodes:")
for max_leaf_nodes in candidate_max_leaf_nodes:
    myTrainMAE = getTrainMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    trainMAE.append(myTrainMAE)
    bestTrain.update({max_leaf_nodes : myTrainMAE})
    print("Max leaf nodes: %d  \t\t Train MAE:  %d" %(max_leaf_nodes, myTrainMAE))

print("\nTest MAE for each nodes:")    
for max_leaf_nodes in candidate_max_leaf_nodes:
    myTestMAE = getTestMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    testMAE.append(myTestMAE)
    bestTest.update({max_leaf_nodes : myTestMAE})
    print("Max leaf nodes: %d  \t\t Test MAE:  %d" %(max_leaf_nodes, myTestMAE))
    
best_tree_size_train  = min(bestTrain, key=bestTrain.get)
best_tree_size_test = min(bestTest, key=bestTest.get)
print("\nTrain MAE for each nodes:", trainMAE)
print("\nTest MAE for each nodes:", testMAE)
print("\nBest regression tree size for train = ", best_tree_size_train)
print("Best regression tree size for test = ", best_tree_size_test)

plt.figure()
plt.plot(candidate_max_leaf_nodes, trainMAE, color="red")
plt.plot(candidate_max_leaf_nodes, testMAE, color="green")
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Regressor")
plt.show()

RGS = DecisionTreeRegressor(max_leaf_nodes=best_tree_size_train, random_state=0)
RGS.fit(train_images,train_labels)
scoreRGS = RGS.score(test_images,test_labels)
print("Score Dicession Tree Regressor = ",scoreRGS)
print("Score Dicession Tree Classifier = ",scoreCLS)

# Disini saya menggunkan data train dan data test yang data pixelnya belum dinormalkan
# Sebelumnya data train dan data test disini telah saya buat terlebih dahulu pada Answer Q5
def getTrainMAE(max_leaf_nodes, train_img, test_img, train_lbl, test_lbl):
    modelTrain = DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    modelTrain.fit(train_img, train_lbl)
    predictTrain = modelTrain.predict(train_img)
    mae = mean_absolute_error(train_lbl, predictTrain)
    return (mae)

def getTestMAE(max_leaf_nodes, train_img, test_img, train_lbl, test_lbl):
    modelTest = DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    modelTest.fit(train_img, train_lbl)
    predictTest = modelTest.predict(test_img)
    mae = mean_absolute_error(test_lbl, predictTest)
    return (mae)

trainMAE = []
testMAE = []

bestTrain = {}
bestTest = {}

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
print("Train MAE for each nodes:")
for max_leaf_nodes in candidate_max_leaf_nodes:
    myTrainMAE = getTrainMAE(max_leaf_nodes, train_img, test_img, train_lbl, test_lbl)
    trainMAE.append(myTrainMAE)
    bestTrain.update({max_leaf_nodes : myTrainMAE})
    print("Max leaf nodes: %d  \t\t Train MAE:  %d" %(max_leaf_nodes, myTrainMAE))
    
print("\nTest MAE for each nodes:")
for max_leaf_nodes in candidate_max_leaf_nodes:
    myTestMAE = getTestMAE(max_leaf_nodes, train_img, test_img, train_lbl, test_lbl)
    testMAE.append(myTestMAE)
    bestTest.update({max_leaf_nodes : myTestMAE})
    print("Max leaf nodes: %d  \t\t Test MAE:  %d" %(max_leaf_nodes, myTestMAE))

best_tree_size_train  = min(bestTrain, key=bestTrain.get)
best_tree_size_test = min(bestTest, key=bestTest.get)
print("\nTrain MAE for each nodes:", trainMAE)
print("Test MAE for each nodes:", testMAE)
print("\nBest tree size for train = ", best_tree_size_train)
print("Best tree size for test = ", best_tree_size_test)

plt.figure()
plt.plot(candidate_max_leaf_nodes, trainMAE, color="red")
plt.plot(candidate_max_leaf_nodes, testMAE, color="green")
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Classifier with Unnormalized Dataset")
plt.show()

CLSuN = DecisionTreeClassifier(max_leaf_nodes=best_tree_size_train, random_state=0)
CLSuN.fit(train_img,train_lbl)
scoreCLSuN = CLSuN.score(test_img,test_lbl)
print("Score Decision Tree Classifier (Unnormalized Dataset) = ", scoreCLSuN)