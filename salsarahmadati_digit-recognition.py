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
from sklearn import svm
import numpy as np
%matplotlib inline
# load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
# now we gonna load the second image, reshape it as matrix then display it
i=1
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
#train_images.iloc[i].describe()
#print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])
a = [12, 4, 18,2,42,9,1,0,11, 7]
i = 12
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
i=4
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
i=18
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
i=2
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
i=42
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
i=9
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
i=1
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
i=0
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
i=11
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
i=7
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
print(np.unique(labeled_images.label))
#i use this to find the index of class

#b = np.array(train_labels)
#c = np.where(b == 9)
#print(c)

    
# create histogram for each class (data merged per class)
# Todo
#print(train_labels.iloc[:5])
#this history is a hist 
data1 = train_images.iloc[7]
data2 = train_images.iloc[10]
data3 = train_images.iloc[15]
data1 = np.array(data1)
data2 = np.array(data2)
#data3 = np.array(data3)
data4 = np.append(data1,data2)
data5 = np.append(data4, data3)
print(len(data5))
plt.hist(data5)
#print(type(labeled_images))
num_class = np.unique(labeled_images.label)
print(num_class)
#for i in range(10):
  # print(len(labeled_images.label[labeled_images.label==i]))

#print(type(images.iloc))
#print(train_labels.iloc[:,0].values)
#count_label = len(train_images)
#print(count_label)
clf = svm.SVC() #model
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# Put your verification code here
# Todo
print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
# Define model
images_model = DecisionTreeRegressor(random_state = 0)
images_model.fit(train_images, train_labels)
img_predict = images_model.predict(test_images)
y = test_labels
mean_absolute_error(img_predict, y)
test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i])
data = train_images.iloc[i]
data = np.array(data)
print(data)
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
#put your code here
from sklearn.tree import DecisionTreeClassifier

def getTrainMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels):
    labeled_model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state = 0)
    labeled_model.fit(train_images, train_labels)
    prediction = labeled_model.predict(train_images)
    mae = mean_absolute_error(prediction, train_labels)
    return(mae)

def getTestMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels):
    labeled_model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    labeled_model.fit(train_images,train_labels)
    prediction = labeled_model.predict(test_images)
    mae = mean_absolute_error(prediction, test_labels)
    return(mae)

trainMae = []
testMae = []
best_train_classifier = {}
best_test_classifier  = {}
print("\nTrain MAE :")
for max_leaf_nodes in [5, 25, 50, 100, 250, 500, 1000]:
    my_train_mae = getTrainMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    trainMae.append(my_train_mae)
    
    best_train_classifier.update({max_leaf_nodes : my_train_mae})
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_train_mae))
    
print("\nTest MAE :")
for max_leaf_nodes in [5, 25, 50, 100, 250, 500, 1000]:
    my_test_mae = getTestMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_test_mae))
    testMae.append(my_test_mae)
    best_test_classifier.update({max_leaf_nodes : my_test_mae})

best_tree_size_train_classifier  = min(best_train_classifier, key=best_train_classifier.get)
best_tree_size_test_classifier = min(best_test_classifier, key=best_test_classifier.get)
print("\nBest tree size for train", best_tree_size_train_classifier)
print("Best tree size for test", best_tree_size_test_classifier)
print("\n")
print(trainMae)
print(testMae)
    
#put your code here
def getTrainMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels):
    labeled_model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    labeled_model.fit(train_images, train_labels)
    prediction = labeled_model.predict(train_images)
    MAE = mean_absolute_error(prediction, train_labels)
    return(MAE)

def getTestMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels):
    labeled_model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    labeled_model.fit(train_images, train_labels)
    prediction = labeled_model.predict(test_images)
    MAE = mean_absolute_error(prediction, test_labels)
    return(MAE)

trainMae = []
testMae = []
best_train_regressor = {}
best_test_regressor  = {}

print("\nTrain MAE :")
for max_leaf_nodes in [5, 25, 50, 100, 250, 500, 1000]:
    my_train_mae = getTrainMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    trainMae.append(my_train_mae)
    best_train_regressor.update({max_leaf_nodes : my_train_mae})
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_train_mae))
    
print("\nTest MAE :")
for max_leaf_nodes in [5, 25, 50, 100, 250, 500, 1000]:
    my_test_mae = getTestMAE(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    testMae.append(my_test_mae)
    best_test_regressor.update({max_leaf_nodes : my_test_mae})
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_test_mae))
    
best_tree_size_train_regressor  = min(best_train_regressor, key=best_train_regressor.get)
best_tree_size_test_regressor = min(best_test_regressor, key=best_test_regressor.get)
print("\nBest tree size for train", best_tree_size_train_regressor)
print("Best tree size for test", best_tree_size_test_regressor)
print("\n")
print(trainMae)
print(testMae)   
#put your code here