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
images = labeled_images.iloc[0:5000,1:] # -> X
labels = labeled_images.iloc[0:5000,:1] # -> y
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
labels.describe()
#train_labels.describe()
#test_labels.describe()
i=1
#print(train_images.iloc[i])
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
a = np.unique(train_labels).tolist() # mencari label 0 - 9 pada train_labels dan mengubah tipe-nya menjadi list
#print(a)
b = train_labels.values.tolist() #mengubah type data train_label menjadi list
#print(b)
arr = []
for k in b:
    for j in k:
       arr.append(j)

#print(arr)
#print(a)
for k in range(len(a)):
    tmp = arr.index(a[k])
    img=train_images.iloc[tmp].values
    img=img.reshape((28,28))
    print("Label:", a[k])
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[tmp,0])
    plt.show()
    #print(tmp)
print("Index:", i)
#print(train_images.iloc[i].values)
#print(type(train_images.iloc[i]))
#print(i)
plt.title(i)
plt.xlabel('panjang pixel')
plt.ylabel('jumlah kolom')
plt.hist(train_images.iloc[i])
count_data = train_labels["label"]
count_data.value_counts()
#create histogram for each class (data merged per class)
# Todo
#print(train_labels.iloc[:5])
data1 = train_images.iloc[1]
data2 = train_images.iloc[3]
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.append(data1,data2)
#print(len(data3))
plt.title('Label 1 dan 3')
plt.xlabel('panjang pixel')
plt.ylabel('jumlah kolom')
plt.hist(data3)
for k in range(len(a)):
    tmp = arr.index(a[k])
    data = train_images.iloc[tmp]
    data = np.array(data)
    print("Label",a[k])
    print("Index",tmp)
    plt.title(a[k])
    plt.xlabel('panjang pixel')
    plt.ylabel('jumlah kolom')
    plt.hist(data)
    plt.show()
S_Data = []
S_Data = np.array(S_Data)
#print(type(S_Data))
#print(S_Data)
for k in range(len(a)):
    tmp = arr.index(a[k])
    data = train_images.iloc[tmp]
    data = np.array(data)
    S_Data = np.append(S_Data, data)

#membuat diagram semua kelas di gabungkan
plt.title('Semua Kelas')
plt.xlabel('panjang pixel')
plt.ylabel('jumlah kolom')
plt.hist(S_Data)
plt.show()
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# Put your verification code here
# Todo
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

model = DecisionTreeRegressor(random_state = 0)
model.fit(train_images, train_labels)
prediction = model.predict(test_images)
mae = mean_absolute_error(prediction, test_labels)

print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number
print(mae)
#print("Index:", i)
test_images[test_images>0] = 1
train_images[train_images>0] = 1

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])

#print(train_images.iloc[i].values)
print("Index",i)
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
from sklearn.model_selection import GridSearchCV
kernels = ['poly', 'rbf', 'linear']
gammas = [0.001, 0.0001]
Cs = [1, 10, 100]

candidate_parameters = {'kernel': kernels, 'gamma': gammas, 'C': Cs}

clf = GridSearchCV(estimator=svm.SVC(), param_grid = candidate_parameters)
clf.fit(train_images,train_labels.values.ravel())
best_kernel = clf.best_estimator_.kernel
best_gamma = clf.best_estimator_.gamma
best_C = clf.best_estimator_.C

print('Best Kernel:',best_kernel)
print('Best C:',best_C) 
print('Best Gamma:',best_gamma)
psv = svm.SVC(C=best_C,kernel=best_kernel, gamma = best_gamma)
psv.fit(train_images, train_labels.values.ravel())
scores = psv.score(test_images,test_labels)
print("SVM:",scores)
from sklearn.tree import DecisionTreeClassifier
def get_mae_train_classifier(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model_train = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state = 0)
    model_train.fit(train_X, train_y)
    predik_train = model_train.predict(train_X)
    mae = mean_absolute_error(train_y, predik_train)
    return(mae)
def get_mae_test_classifier(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model_test = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state = 0)
    model_test.fit(train_X, train_y)
    predik_test = model_test.predict(val_X)
    mae = mean_absolute_error(val_y, predik_test)
    return(mae)
train_mae_classifier = []
test_mae_classifier = []

best_train_classifier = {}
best_test_classifier = {}
#train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae_train = get_mae_train_classifier(max_leaf_nodes = max_leaf_nodes, train_X = train_images, val_X = test_images, train_y = train_labels, val_y = test_labels)
    train_mae_classifier.append(my_mae_train)
    best_train_classifier.update({max_leaf_nodes : my_mae_train})
    my_mae_test = get_mae_test_classifier(max_leaf_nodes = max_leaf_nodes, train_X = train_images, val_X = test_images, train_y = train_labels, val_y = test_labels)
    test_mae_classifier.append(my_mae_test)
    best_test_classifier.update({max_leaf_nodes : my_mae_test})

    
best_tree_size_train_classifier  = min(best_train_classifier, key=best_train_classifier.get)
best_tree_size_test_classifier = min(best_test_classifier, key=best_test_classifier.get)
print("Best tree size for train", best_tree_size_train_classifier)
print("Best tree size for test", best_tree_size_test_classifier)
print(train_mae_classifier)
print(test_mae_classifier)

plt.figure()
plt.plot(candidate_max_leaf_nodes, train_mae_classifier, color="blue")
plt.plot(candidate_max_leaf_nodes, test_mae_classifier, color="red")
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Classifier")
plt.show()
CLS = DecisionTreeClassifier(max_leaf_nodes=best_tree_size_train_classifier, random_state=0)
CLS.fit(train_images,train_labels)
scoreCLS = CLS.score(test_images,test_labels)
print("Score DTC:",scoreCLS)
print("Score SVM:", scores)
#Decisiion Tree Regresor
def get_mae_train(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model_train = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model_train.fit(train_X, train_y)
    predik_train = model_train.predict(train_X)
    mae = mean_absolute_error(train_y, predik_train)
    return(mae)
def get_mae_test(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model_test = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model_test.fit(train_X, train_y)
    predik_test = model_test.predict(val_X)
    mae = mean_absolute_error(val_y, predik_test)
    return(mae)
train_mae = []
test_mae = []

best_train = {}
best_test = {}
#train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae_train = get_mae_train(max_leaf_nodes = max_leaf_nodes, train_X = train_images, val_X = test_images, train_y = train_labels, val_y = test_labels)
    train_mae.append(my_mae_train)
    best_train.update({max_leaf_nodes : my_mae_train})
    my_mae_test = get_mae_test(max_leaf_nodes = max_leaf_nodes, train_X = train_images, val_X = test_images, train_y = train_labels, val_y = test_labels)
    test_mae.append(my_mae_test)
    best_test.update({max_leaf_nodes : my_mae_test})

best_tree_size_train  = min(best_train, key=best_train.get)
best_tree_size_test = min(best_test, key=best_test.get)
print("Best tree size for train", best_tree_size_train)
print("Best tree size for test", best_tree_size_test)
print(train_mae)
print(test_mae)
plt.figure()
plt.plot(candidate_max_leaf_nodes, train_mae, color="blue")
plt.plot(candidate_max_leaf_nodes, test_mae, color="red")
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Regression")
plt.show()
RGS = DecisionTreeRegressor(max_leaf_nodes=best_tree_size_train, random_state=0)
RGS.fit(train_images,train_labels)
scoreRGS = RGS.score(test_images,test_labels)
print("Score DTR:",scoreRGS)
print("Score DTC:",scoreCLS)
print("Score SVM:",scores)
# load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:] # -> X
labels = labeled_images.iloc[0:5000,:1] # -> y
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
def get_mae_train_classifier(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model_train = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state = 0)
    model_train.fit(train_X, train_y)
    predik_train = model_train.predict(train_X)
    mae = mean_absolute_error(train_y, predik_train)
    return(mae)

def get_mae_test_classifier(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model_test = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state = 0)
    model_test.fit(train_X, train_y)
    predik_test = model_test.predict(val_X)
    mae = mean_absolute_error(val_y, predik_test)
    return(mae)
train_mae_classifier = []
test_mae_classifier = []

best_train_classifier = {}
best_test_classifier = {}
#train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae_train = get_mae_train_classifier(max_leaf_nodes = max_leaf_nodes, train_X = train_images, val_X = test_images, train_y = train_labels, val_y = test_labels)
    train_mae_classifier.append(my_mae_train)
    best_train_classifier.update({max_leaf_nodes : my_mae_train})
    my_mae_test = get_mae_test_classifier(max_leaf_nodes = max_leaf_nodes, train_X = train_images, val_X = test_images, train_y = train_labels, val_y = test_labels)
    test_mae_classifier.append(my_mae_test)
    best_test_classifier.update({max_leaf_nodes : my_mae_test})

    
best_tree_size_train_classifier  = min(best_train_classifier, key=best_train_classifier.get)
best_tree_size_test_classifier = min(best_test_classifier, key=best_test_classifier.get)
print("Best tree size for train", best_tree_size_train_classifier)
print("Best tree size for test", best_tree_size_test_classifier)
print(train_mae_classifier)
print(test_mae_classifier)
plt.figure()
plt.plot(candidate_max_leaf_nodes, train_mae_classifier, color="blue")
plt.plot(candidate_max_leaf_nodes, test_mae_classifier, color="red")
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Classifier (unnormalized dataset)")
plt.show()