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
%matplotlib inline
print("setup complete")
# load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
print("success")
# now we gonna load the second image, reshape it as matrix than display it
i=1
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
# put your code
# Find unique labels from train_labels and stored the index of unique labels into array.
unikIndex = train_labels.drop_duplicates(subset='label')
unikIndex = unikIndex.sort_values(by=['label'])
unikIndex = unikIndex.index.values
eachClass = []
a, b = 0, 0
while len(eachClass) < len(unikIndex):
    if train_labels.index[a] == unikIndex[b]:
        eachClass.append(a)
        a = 0
        b += 1
    else:
        a += 1

# Plot images in eachClass array.
for j in eachClass:
    plt.figure()
    img = train_images.iloc[j].values
    img = img.reshape((28,28))
    plt.imshow(img,cmap='binary')
    title = "Class " + str(train_labels.iloc[j,0])
    plt.title(title)
#print(train_images.iloc[i].describe())
#print(type(train_images.iloc[i]))
plt.figure(i)
plt.hist(train_images.iloc[i])
title = "Class " + str(train_labels.iloc[i,0])
plt.title(title)
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
for j in eachClass:
    plt.figure(j)
    plt.hist(train_images.iloc[j])
    title = "Class " + str(train_labels.iloc[j,0])
    plt.title(title)
# create histogram for each class (data merged per class)
# Todo
data0 = np.array(train_images.iloc[eachClass[0]])
data1 = np.array(train_images.iloc[eachClass[1]])
data2 = np.array(train_images.iloc[eachClass[2]])
data3 = np.array(train_images.iloc[eachClass[3]])
data4 = np.array(train_images.iloc[eachClass[4]])
data5 = np.array(train_images.iloc[eachClass[5]])
data6 = np.array(train_images.iloc[eachClass[6]])
data7 = np.array(train_images.iloc[eachClass[7]])
data8 = np.array(train_images.iloc[eachClass[8]])
data9 = np.array(train_images.iloc[eachClass[9]])

dataTotal = np.append(data0, data1)
dataTotal2 = np.append(dataTotal, data2)
dataTotal3 = np.append(dataTotal2, data3)
dataTotal4 = np.append(dataTotal3, data4)
dataTotal5 = np.append(dataTotal4, data5)
dataTotal6 = np.append(dataTotal5, data6)
dataTotal7 = np.append(dataTotal6, data7)
dataTotal8 = np.append(dataTotal7, data8)
dataTotal9 = np.append(dataTotal8, data9)
#print(len(dataTotal9))
plt.hist(dataTotal9)
plt.title("Histogram with one sample from each class in each class")
# Count the frequency a value on 'label' colums
frequencyEachClass = train_labels.groupby('label').size()
print(frequencyEachClass)
for m in range (len(eachClass)):
    temp = []
    for n in range(len(train_images)):
       if (train_labels.iloc[n].label == m):
            img = train_images.iloc[n].values
            temp.extend(img)
    #print(len(temp))
    plt.figure(m)
    plt.hist(temp)
    title = "Historgram with all sample in class " + str(m)
    plt.title(title)

#print(train_labels.values)
#print(train_labels.values.ravel())
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
#print(type(clf))
# Put your verification code here
# Todo
from sklearn.metrics import mean_absolute_error
test_predict = clf.predict(test_images)
mae = mean_absolute_error(test_labels, test_predict)
print(mae)
#print(train_labels.values.ravel())
#print(np.unique(test_labels)) # to see class number 
test_images[test_images>0]=1
train_images[train_images>0]=1
#print(test_images.describe())

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
# now plot again the histogram
plt.hist(train_images.iloc[i])
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# MAE of retrain model
test_predict = clf.predict(test_images)
mae = mean_absolute_error(test_labels, test_predict)
print(mae)
# My Experiment to find another score
#cap=[1, 2, 3, 4, 5, 6, 500]
#for i in cap:
 #   test_images[test_images>0]=i
  #  train_images[train_images>0]=i
   # clf = svm.SVC()
    #clf.fit(train_images, train_labels.values.ravel())
    #score = clf.score(test_images,test_labels)
    #print("cap:",i,"score:",score)
# back to the rule
#test_images[test_images>0]=1
#train_images[train_images>0]=1
#clf = svm.SVC()
#clf.fit(train_images, train_labels.values.ravel())
#print(clf.score(test_images,test_labels))
# Test again to data test
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
#results=clf.predict(test_data[0:5000])
results=clf.predict(test_data)
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
test_images[test_images>0]=1
train_images[train_images>0]=1
clf = svm.SVC(gamma=20.0)
clf.fit(train_images, train_labels.values.ravel())
test_predict = clf.predict(test_images)
print("MAE:", mean_absolute_error(test_labels, test_predict))
print("Score:", clf.score(test_images, test_labels),"\n")

clf = svm.SVC(kernel='linear')
clf.fit(train_images, train_labels.values.ravel())
test_predict = clf.predict(test_images)
print("MAE:", mean_absolute_error(test_labels, test_predict))
print("Score:", clf.score(test_images, test_labels),"\n")

clf = svm.SVC(C = 2.0)
clf.fit(train_images, train_labels.values.ravel())
test_predict = clf.predict(test_images)
print("MAE:", mean_absolute_error(test_labels, test_predict))
print("Score:", clf.score(test_images, test_labels),"\n")

from sklearn.model_selection import GridSearchCV as GSCV
kernels = ['poly','rbf','linear']
gammas = [0.001, 0.0001]
Cs = [1, 10, 100]

candidate_parameters = {'kernel':kernels, 'gamma':gammas, 'C':Cs}

clf = GSCV(estimator=svm.SVC(), param_grid=candidate_parameters)
clf.fit(train_images, train_labels.values.ravel())
best_kernel = clf.best_estimator_.kernel
best_gamma = clf.best_estimator_.gamma
best_C = clf.best_estimator_.C

print("Best Kernel:", best_kernel)
print("Best C:", best_C)
print("Best Gamma", best_gamma)
from sklearn.tree import DecisionTreeClassifier

#print(train_images.iloc[0].values) #check the data. Is data normalized or unnormalized.
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(train_images, train_labels)
dtc_pred = dtc.predict(test_images)
print("MAE:", mean_absolute_error(test_labels, dtc_pred))
print("Score:", dtc.score(test_images, test_labels))
dtc = DecisionTreeClassifier(random_state=0, max_depth = 10)
dtc.fit(train_images, train_labels)
dtc_pred = dtc.predict(test_images)
print("MAE:", mean_absolute_error(test_labels, dtc_pred))
print("Score:", dtc.score(test_images, test_labels))
dtc = DecisionTreeClassifier(random_state=0, max_leaf_nodes=400)
dtc.fit(train_images, train_labels)
dtc_pred = dtc.predict(test_images)
print("MAE:", mean_absolute_error(test_labels, dtc_pred))
print("Score:", dtc.score(test_images, test_labels))
# Candidate Array Controller
candidate_leaf_nodes=[5, 10, 20, 50, 70, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#candidate_leaf_nodes=[]
#for i in range(50):
 #   i = (i+2)*20
  #  candidate_leaf_nodes.append(i)
print("success")
def get_test_mae_dtc(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

def get_train_mae_dtc(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_train = model.predict(train_X)
    mae = mean_absolute_error(train_y, preds_train)
    return(mae)

my_test_mae_dtc = []
my_train_mae_dtc = []

train_X, val_X, train_y, val_y = train_test_split(train_images, train_labels)
for max_leaf_nodes in candidate_leaf_nodes:
    test_mae = get_test_mae_dtc(max_leaf_nodes, train_X, val_X, train_y, val_y)
    my_test_mae_dtc.append(test_mae)
    train_mae = get_train_mae_dtc(max_leaf_nodes, train_X, val_X, train_y, val_y)
    my_train_mae_dtc.append(train_mae)

plt.figure()
plt.plot(candidate_leaf_nodes, my_test_mae_dtc, color="black")
plt.plot(candidate_leaf_nodes, my_train_mae_dtc, color="red")
plt.ylabel("MAE")
plt.xlabel("Tree Depth")
plt.title("Overfitting and Underfitting")
# I used candidate_leaf_nodes before
def best_leaf_nodes(leaf_nodes, train_X, test_X, train_y, test_y):
    memory = {}
    for max_leaf_nodes in leaf_nodes:
        my_mae = get_test_mae_dtc(max_leaf_nodes, train_X, test_X, train_y, test_y)
        temp = {max_leaf_nodes:my_mae}
        memory.update(temp)
        print("max leaf nodes:",max_leaf_nodes,"\t\tMAE:", my_mae)
    best_leaf = min(memory, key=memory.get)
    return best_leaf

bestLeaf_dtc = best_leaf_nodes(candidate_leaf_nodes, train_images, test_images, train_labels, test_labels)
print("best max leaf nodes:",bestLeaf_dtc)
dtc = DecisionTreeClassifier(max_leaf_nodes=bestLeaf_dtc, random_state=0)
dtc.fit(train_images, train_labels)
testPrediction = dtc.predict(test_images)
print("max leaf nodes:",bestLeaf_dtc)
print("MAE with max leaf nodes:", mean_absolute_error(test_labels, testPrediction))
print("Score:",dtc.score(test_images, test_labels))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(train_images, train_labels)
dtr_pred = dtr.predict(test_images)
print("MAE:", mean_absolute_error(test_labels, dtr_pred))
print("Score:", dtr.score(test_images, test_labels))
# I used candidate array before

def get_test_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

def get_train_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_train = model.predict(train_X)
    mae = mean_absolute_error(train_y, preds_train)
    return(mae)

my_test_mae = []
my_train_mae = []

train_X, val_X, train_y, val_y = train_test_split(train_images, train_labels)
for max_leaf_nodes in candidate_leaf_nodes:
    test_mae = get_test_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    my_test_mae.append(test_mae)
    train_mae = get_train_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    my_train_mae.append(train_mae)

plt.figure()
plt.plot(candidate_leaf_nodes, my_test_mae, color="black")
plt.plot(candidate_leaf_nodes, my_train_mae, color="red")
plt.ylabel("MAE")
plt.xlabel("Tree Depth")
plt.title("Overfitting and Underfitting")
def best_leaf_nodes(leaf_nodes, train_X, test_X, train_y, test_y):
    memory = {}
    for max_leaf_nodes in leaf_nodes:
        my_mae = get_test_mae(max_leaf_nodes, train_X, test_X, train_y, test_y)
        temp = {max_leaf_nodes:my_mae}
        memory.update(temp) 
        print("max leaf nodes:",max_leaf_nodes,"\t\tMAE:", my_mae)
    best_leaf = min(memory, key=memory.get)
    return best_leaf

bestLeaf = best_leaf_nodes(candidate_leaf_nodes, train_images, test_images, train_labels, test_labels)
print("best max leaf nodes:",bestLeaf)
dtr = DecisionTreeRegressor(max_leaf_nodes=bestLeaf, random_state=0)
dtr.fit(train_images, train_labels)
testPrediction = dtr.predict(test_images)
print("max leaf nodes:",bestLeaf)
print("MAE with max leaf nodes:", mean_absolute_error(test_labels, testPrediction))
print("Score:",dtr.score(test_images, test_labels))
# load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
print("success")
#print(train_images.iloc[0].values) #check the data. Is data normalized or unnormalized.
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(train_images, train_labels)
dtc_pred = dtc.predict(test_images)
print("MAE:", mean_absolute_error(test_labels, dtc_pred))
print("Score:", dtc.score(test_images, test_labels))
def get_test_mae_dtc(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

def get_train_mae_dtc(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_train = model.predict(train_X)
    mae = mean_absolute_error(train_y, preds_train)
    return(mae)

my_test_mae_dtc = []
my_train_mae_dtc = []

train_X, val_X, train_y, val_y = train_test_split(train_images, train_labels)
for max_leaf_nodes in candidate_leaf_nodes:
    test_mae = get_test_mae_dtc(max_leaf_nodes, train_X, val_X, train_y, val_y)
    my_test_mae_dtc.append(test_mae)
    train_mae = get_train_mae_dtc(max_leaf_nodes, train_X, val_X, train_y, val_y)
    my_train_mae_dtc.append(train_mae)

plt.figure()
plt.plot(candidate_leaf_nodes, my_test_mae_dtc, color="black")
plt.plot(candidate_leaf_nodes, my_train_mae_dtc, color="red")
plt.ylabel("MAE")
plt.xlabel("Tree Depth")
plt.title("Overfitting and Underfitting")
# I used candidate_leaf_nodes before
def best_leaf_nodes(leaf_nodes, train_X, test_X, train_y, test_y):
    memory = {}
    for max_leaf_nodes in leaf_nodes:
        my_mae = get_test_mae_dtc(max_leaf_nodes, train_X, test_X, train_y, test_y)
        temp = {max_leaf_nodes:my_mae}
        memory.update(temp)
        print("max leaf nodes:",max_leaf_nodes,"\t\tMAE:", my_mae)
    best_leaf = min(memory, key=memory.get)
    return best_leaf

bestLeaf_dtc = best_leaf_nodes(candidate_leaf_nodes, train_images, test_images, train_labels, test_labels)
print("best max leaf nodes:",bestLeaf_dtc)
dtc = DecisionTreeClassifier(max_leaf_nodes=bestLeaf_dtc, random_state=0)
dtc.fit(train_images, train_labels)
testPrediction = dtc.predict(test_images)
print("max leaf nodes:",bestLeaf_dtc)
print("MAE with max leaf nodes:", mean_absolute_error(test_labels, testPrediction))
print("Score:",dtc.score(test_images, test_labels))