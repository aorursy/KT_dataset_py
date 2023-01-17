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
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
#%matplotlib inline
print('Import complete.')
# load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:] #return a DataFrame consisted of rows indexed in 0:5000, and columns indexed from index 1.
labels = labeled_images.iloc[0:5000,:1] #return a DataFrame consisted of rows indexed in 0:5000, and columns indexed from beginning to index 1
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
# Data check
print(labeled_images.describe())
print(images.describe())
print(labels.describe())
i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
number = np.unique(train_labels) # np.unique is for returning the unique element of given array
print (number)


index = [0,0,0,0,0,0,0,0,0,0]    # declare temporary index for searching the result in train_labels, 
                                 # this index will eventualy be replace with the number location in every class from 0-9

# In order to find said Index, we use loops in every row in train_labels 

for i in range (len(train_labels)):
    label = train_labels.iloc[i].label # label variable contain number class, in order to find the index of i
    index[label] = i                   # (iloc[i] serach the row of i, and .label search the coloumn of label)
    
for i in index:     # Now we print the image for every column of number
    plt.figure()
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[i,0])

#train_images.iloc[i].describe()
#print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])
labelcount = train_labels["label"]
print(labelcount.value_counts())
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

label = [[],[],[],[],[],[],[],[],[],[]]
for j in range(10):
    for i in range(len(train_images)):
        if (train_labels.iloc[i].label == j):
            data = train_images.iloc[i]
            data = np.array(data)
            label[j] = np.append(label[j],data)
            
    plt.figure(j)
    plt.hist(label[j])
    plt.title(j)
clf = svm.SVC() # Define model
clf.fit(train_images, train_labels.values.ravel()) # Fit: Capture patterns from provided data.
clf.score(test_images,test_labels) # Determine how accurate the model's
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state=0)
tree.fit(train_images, train_labels)
test_predict = tree.predict(test_images)
print(mean_absolute_error(test_labels, test_predict))
print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number

for y in index:
    plt.figure(y)
    img=train_images.iloc[y].values
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[y,0])
test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
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
from sklearn.svm import SVC

# Set the parameters by cross-validation
parameters = {'gamma': [0.01, 0.001, 0.0001],'C': [1, 10, 100,1000]}

# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid = parameters)
clf.fit(train_images,train_labels.values.ravel())
print('Best C:',clf.best_estimator_.C) 
print('Best Gamma:',clf.best_estimator_.gamma)
#final svm
best_c = 10
best_gamma = 0.01
clf_final = svm.SVC(C=best_c,gamma=best_gamma)
clf_final.fit(train_images, train_labels.values.ravel())
finalsvm = clf_final.score(test_images,test_labels)
print(clf_final.score(test_images,test_labels))
from sklearn.tree import DecisionTreeClassifier
def get_mae_train_classifie(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_train = model.predict(train_X)
    mae = mean_absolute_error(train_y, preds_train)
    return(mae)

def get_mae_test_classifie(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


maee_train=[]
maee_test= []
leaf_nodes=[5,25,50,70,100,300,500,1000,3000,5000,7000]
for max_leaf_nodes in leaf_nodes:
    my_maetrain = get_mae_train_classifie(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    my_maetest = get_mae_test_classifie(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    maee_train.append(my_maetrain)
    maee_test.append(my_maetest)

plt.figure()
plt.plot(leaf_nodes,maee_test,color="red",label='Validation')
plt.plot(leaf_nodes,maee_train,color="blue",label='Training')
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Classifier")
plt.legend()
plt.show()

print (maee_train)
print (maee_test)
#final model dtclassifier

best_tree_size=1000
treeclassifie = DecisionTreeClassifier(max_leaf_nodes=best_tree_size, random_state=0)
treeclassifie.fit(train_images,train_labels)
scoredtc = treeclassifie.score(test_images,test_labels)
print ("DTC = ",scoredtc)
print ("SVM = ",finalsvm)
# Decision Tree Regressor
def get_mae_train_regress(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    preds_train = model.predict(train_X)
    mae = mean_absolute_error(train_y, preds_train)
    return(mae)

def get_mae_test_regress(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

maee_train=[]
maee_test= []
leaf_nodes=[5,25,50,70,100,300,500,1000,3000,5000,7000]
for max_leaf_nodes in leaf_nodes:
    my_maetrain = get_mae_train_regress(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    my_maetest = get_mae_test_regress(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    maee_train.append(my_maetrain)
    maee_test.append(my_maetest)

plt.figure()
plt.plot(leaf_nodes,maee_test,color="red",label='Validation')
plt.plot(leaf_nodes,maee_train,color="blue",label='Training')
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

print (maee_train)
print (maee_test)
#final model dtregressor
best_tree_size=1000
treeregres = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
treeregres.fit(train_images,train_labels)
scoredtr = treeregres.score(test_images,test_labels)
print ("DTR = ",scoredtr)
print ("DTC = ",scoredtc)
print ("SVM = ",finalsvm)
# Decision Tree Classifier
# labeled_images.iloc[0:5000,1:], yang terseleksi kedalam variabel images baris ke-0 sampai 4999 dan kolom ke-1 sampai kolom terakhir
images = labeled_images.iloc[0:5000,1:]
# labeled_images.iloc[0:5000,:1], yang terseleksi kedalam variabel labels baris ke-0 sampai 4999 dan kolom ke-0
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
best_tree_size=1000
tree = DecisionTreeClassifier(max_leaf_nodes=best_tree_size,random_state=0)
tree.fit(train_images, train_labels)
test_predict = tree.predict(test_images)
#print(mean_absolute_error(test_labels, test_predict))
scoredtr_ver2 = tree.score(test_images,test_labels)
print ("DTC = ",scoredtr_ver2)