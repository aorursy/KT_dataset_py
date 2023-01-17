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
point = train_labels.values.tolist()
#Print(point)
array = []
for n in point:
    for a in n:
        array.append(a)
        
    
label = np.unique(train_labels).tolist()
#print(label)
for n in range(len(label)):
    nilai = array.index(label[n])
    img=train_images.iloc[nilai].values
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    print("index", nilai)
    plt.title(train_labels.iloc[nilai,0])
    plt.show()

train_images.iloc[i].describe()
#print(type(train_images.iloc[i]))
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
#plt.title("label 1 dan 3")
plt.hist(data3)
for i in range(len(label)):
    nilai = array.index(label[i])
    data  = train_images.iloc[nilai]
    data_label = np.array(data)
    plt.title(label[i])
    plt.hist(data)
    plt.show()

    
a = []
all_data = np.array(a)
for k in range(len(label)):
    nilai = array.index(label[k])
    data  = train_images.iloc[nilai]
    data_label = np.array(data)
    all_data = np.append(all_data, data)
plt.title("All Data")
plt.hist(all_data)
plt.show()

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

# Put your verification code here
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

model = DecisionTreeRegressor(random_state = 0)
model.fit(train_images, train_labels)
prediction = model.predict(test_images)
mae = mean_absolute_error(prediction, test_labels)

# Todo
print("maenya adalah ", mae)
print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number

i = 1
print (i)
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
from sklearn.tree import DecisionTreeRegressor
def get_mae_train(max_leaf_nodes, train_x, val_x, train_y, val_y):
    train_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    train_model.fit(train_x,train_y)
    predictions = train_model.predict(train_x)
    mae = mean_absolute_error(predictions,train_y)
    return(mae)

def get_mae_test(max_leaf_nodes, train_x, val_x, train_y, val_y):
    test_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    test_model.fit(train_x,train_y)
    predictions = test_model.predict(val_x)
    mae = mean_absolute_error(predictions,val_y)
    return(mae)
    

best_train = {}
best_test = {}
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500,1000,2500,5000,10000]
train=[]
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae_train = get_mae_train(max_leaf_nodes = max_leaf_nodes, train_x = train_images, val_x = test_images, train_y = train_labels, val_y = test_labels)
    train.append(my_mae_train)
    best_train.update({max_leaf_nodes : my_mae_train})

test = []
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae_test = get_mae_test(max_leaf_nodes= max_leaf_nodes, train_x=train_images, val_x=test_images, train_y=train_labels, val_y=test_labels)
    test.append(my_mae_test)
    best_test.update({max_leaf_nodes : my_mae_test})
    
    
 
best_tree_size_train  = min(best_train, key=best_train.get)
best_tree_size_test = min(best_test, key=best_test.get)
print(best_tree_size_train)
print(best_tree_size_test)
print(train)
print(test)
plt.figure()
plt.plot(candidate_max_leaf_nodes, train, color="blue")
plt.plot(candidate_max_leaf_nodes, test, color="red")
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Regression")
plt.show()
from sklearn.tree import DecisionTreeClassifier
def get_mae_train_Classifier(max_leaf_nodes, train_x, val_x, train_y, val_y):
    train_model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    train_model.fit(train_x,train_y)
    predictions = train_model.predict(train_x)
    mae = mean_absolute_error(predictions,train_y)
    return(mae)

def get_mae_test_Classifier(max_leaf_nodes, train_x, val_x, train_y, val_y):
    test_model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    test_model.fit(train_x,train_y)
    predictions = test_model.predict(val_x)
    mae = mean_absolute_error(predictions,val_y)
    return(mae)
C_train = {}
C_test = {}
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500,1000,2500,5000,10000]
train_Classifier=[]
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae_train = get_mae_train_Classifier(max_leaf_nodes = max_leaf_nodes, train_x = train_images, val_x = test_images, train_y = train_labels, val_y = test_labels)
    train_Classifier.append(my_mae_train)
    C_train.update({max_leaf_nodes : my_mae_train})

test_Classifier = []
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae_test = get_mae_test_Classifier(max_leaf_nodes= max_leaf_nodes, train_x=train_images, val_x=test_images, train_y=train_labels, val_y=test_labels)
    test_Classifier.append(my_mae_test)
    C_test.update({max_leaf_nodes : my_mae_test})
    
tree_size_train_Classifier  = min(C_train, key=C_train.get)
tree_size_test_Classifier = min(C_test, key=C_test.get)
print(tree_size_train_Classifier)
print(tree_size_test_Classifier)
print(train_Classifier)
print(test_Classifier) 
plt.figure()
plt.plot(candidate_max_leaf_nodes, train_Classifier, color="blue")
plt.plot(candidate_max_leaf_nodes, test_Classifier, color="red")
plt.ylabel("MAE")
plt.xlabel("Tree Depth")
plt.title("Decision Tree Classifier")

plt.show()
from sklearn.svm import SVC





