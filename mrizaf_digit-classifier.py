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
images = labeled_images.iloc[0:5000,1:]  # get 5000 rows data, and all columns data (except the first) (features)
labels = labeled_images.iloc[0:5000,:1]  # get 5000 rows data, and first column data (target)
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
# now we gonna load the second image, reshape it as matrix than display it
i=1
img=train_images.iloc[i].values  # get array of row i data of train_images
img=img.reshape((28,28))  # reshape the array to (28x28)
plt.imshow(img,cmap='gray')  #  plot the data of img
plt.title(train_labels.iloc[i,0])  # give title to the plot
# create empty array with 10 spot for each class number (0-9),
# to save all the row index of each class from train_labels
class_n = [[] for i in range(len(np.unique(test_labels)))] 

for i in range(len(train_labels)):  # loop through the row length of train_labels data
    n = train_labels.iloc[i, 0]  # get the data (class number) in the i row and first column
    class_n[n] += [i]  # expand the class_n index n (n=class number) with i (i=index of row)

# create empty array for image of each class number, to save image data of each class
imgs = []
class_ni = []

for i in range(len(class_n)):  # loop through the length of class_n
    n = class_n[i][0]  # get the first data of i from class_n
    class_ni += [n]
    img = train_images.iloc[n].values  # get array of row n data of train_images
    img = img.reshape((28, 28))  # reshape the array to (28x28)
    imgs += [img]  # expand the imgs with img

# for making join image of each class
#img = np.concatenate((imgs), axis=1)  # join all arrays in imgs with axis=1 (increase the column)
#plt.imshow(img, cmap='gray')  # plot the data of img
#plt.title("Image for each class")  # give title to the plot

# for making separate image of each class
def imshow_plot(ax, data, col, title):
    ax.set_title(title)
    ax.imshow(data, cmap=col)

fig, axs = plt.subplots(2, 5, figsize=(8, 4))
fig.suptitle('Images for each class', y=1.05, fontsize=16)

n = 0
for ax in axs.flatten():
    imshow_plot(ax, imgs[n], 'gray', "Class " + str(n))
    n += 1

plt.tight_layout()
plt.show()
plt.close()
i=1
#train_images.iloc[i].describe()
#print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])

#print(train_labels.iloc[i])
# create histogram for each class (data merged per class)
# Todo
#print(train_labels.iloc[:5])
data1 = train_images.iloc[1]
data2 = train_images.iloc[3]
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.append(data1,data2)
#print(len(data3))
plt.hist(data3)

# code to create histogram for each class 
def hist_plot(ax, data, title):
    ax.set_title(title)
    ax.set_xlabel('pixel')
    ax.set_ylabel('total')
    ax.hist(data)

fig, axs = plt.subplots(5, 2, figsize=(12, 16))
fig.suptitle('Histograms for each class', y=1.025, fontsize=16)

n = 0
for ax in axs.flatten():
    hist_plot(ax, train_images.iloc[class_ni[n]].values, "Class " + str(n))
    n += 1

plt.tight_layout()
plt.show()
plt.close()

# code to create merge histogram of each class
fig = plt.figure()
data1 = np.array(train_images.iloc[class_ni[0]])
data2 = np.array(train_images.iloc[class_ni[1:]])
datan = np.append(data1, (data2))
plt.title("Histogram of each class (data merged)")
plt.xlabel('pixel')
plt.ylabel('total')
plt.hist(datan)
plt.show()
plt.close()
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# Put your verification code here
# Todo
print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number

# code to find the MAE from the model
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

mae = DecisionTreeRegressor()
mae.fit(train_images, train_labels)
mean_absolute_error(test_labels, mae.predict(test_images))
#mean_absolute_error(mae.predict(test_images), test_labels)
i=1
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

def time_call(fn):
    """
    Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    t0 = time()
    fn
    t1 = time()
    return t1-t0
    pass

time1 = 0
time2 = 0
fig = plt.figure(figsize=(20, 5))
for i in range(10):
    data1 = np.random.randint(255, size=(28, 28))
    data2 = data1 % 2
    
    plt.subplot(2, 10, i+1)
    plt.title('Uncapped pixel')
    time1 += time_call(plt.imshow(data1, cmap='binary'))

    plt.subplot(2, 10, i+11)
    plt.title('Capped pixel')
    time2 += time_call(plt.imshow(data2, cmap='binary'))

print("Time to plot 10 uncapped pixel : " + str(time1*1000000) + "\nTime to plot 10 capped pixel : " + str(time2*1000000))
plt.tight_layout()
plt.show()
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
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
n_scores = []
#n_scores = [[] for i in range(len(kernels))]

#for i in range(1, 10, 1):
    #n = 0
for k in kernels:
    clf = svm.SVC(kernel=k, random_state=0)
    #clf = svm.SVC(C=i/10, kernel=k, random_state=0)
    clf.fit(train_images, train_labels.values.ravel())
    n_scores += [clf.score(test_images,test_labels)]
    #n_scores[n] += [clf.score(test_images,test_labels)]
    #n += 1

print(n_scores)
print("SVM:", max(n_scores))
from sklearn.tree import DecisionTreeClassifier

def get_mae_train_test_dtc(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    train_predictions = model.predict(train_X)
    train_mae = mean_absolute_error(train_y, train_predictions)
    test_predictions = model.predict(val_X)
    test_mae = mean_absolute_error(val_y, test_predictions)
    return(train_mae, test_mae)

train_maes = []
test_maes = []

candidate_max_leaf_nodes = [5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000]

for max_leaf_nodes in candidate_max_leaf_nodes:
    train, test = get_mae_train_test_dtc(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    train_maes.append(train)
    test_maes.append(test)
    
best_tree_size = candidate_max_leaf_nodes[test_maes.index(min(test_maes))]
best = "Best tree size for test " + str(best_tree_size)

plt.plot(candidate_max_leaf_nodes, test_maes, color="red", label="Test")
plt.plot(candidate_max_leaf_nodes, train_maes, color="blue", label="Train")
plt.axvline(x=best_tree_size, linewidth=0.5, color="gray", label=best)
plt.title("Decision Tree Classifier (DTC)")
plt.ylabel("Mean Average Error")
plt.xlabel("Tree Depth")
plt.legend(bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0.)
plt.show()

dtc = DecisionTreeClassifier(max_leaf_nodes=best_tree_size, random_state=0)
dtc.fit(train_images,train_labels)
s_dtc = dtc.score(test_images,test_labels)
m_dtc = mean_absolute_error(test_labels, mae.predict(test_images))
print("DTC Score:", s_dtc, "\tDTC MAE:", m_dtc)
print("SVM Score:", max(n_scores))
from sklearn.tree import DecisionTreeRegressor

def get_mae_train_test_dtr(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    train_predictions = model.predict(train_X)
    train_mae = mean_absolute_error(train_y, train_predictions)
    test_predictions = model.predict(val_X)
    test_mae = mean_absolute_error(val_y, test_predictions)
    return(train_mae, test_mae)

train_maes = []
test_maes = []

for max_leaf_nodes in candidate_max_leaf_nodes:
    train, test = get_mae_train_test_dtr(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    train_maes.append(train)
    test_maes.append(test)
    
best_tree_size = candidate_max_leaf_nodes[test_maes.index(min(test_maes))]
best = "Best tree size for test " + str(best_tree_size)

plt.plot(candidate_max_leaf_nodes, test_maes, color="red", label="Test")
plt.plot(candidate_max_leaf_nodes, train_maes, color="blue", label="Train")
plt.axvline(x=best_tree_size, linewidth=0.5, color="gray", label=best)
plt.title("Decision Tree Regressor (DTR)")
plt.ylabel("Mean Average Error")
plt.xlabel("Tree Depth")
plt.legend(bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0.)
plt.show()

dtr = DecisionTreeClassifier(max_leaf_nodes=best_tree_size, random_state=0)
dtr.fit(train_images,train_labels)
s_dtr = dtr.score(test_images,test_labels)
m_dtr = mean_absolute_error(test_labels, mae.predict(test_images))
print("DTR Score:", s_dtr, "\tDTR MAE:", m_dtr)
print("DTC Score:", s_dtc, "\tDTC MAE:", m_dtc)
print("SVM Score:", max(n_scores))
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)

train_maes = []
test_maes = []

for max_leaf_nodes in candidate_max_leaf_nodes:
    train, test = get_mae_train_test_dtc(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    train_maes.append(train)
    test_maes.append(test)
    
best_tree_size = candidate_max_leaf_nodes[test_maes.index(min(test_maes))]
best = "Best tree size for test " + str(best_tree_size)

plt.plot(candidate_max_leaf_nodes, test_maes, color="red", label="Test")
plt.plot(candidate_max_leaf_nodes, train_maes, color="blue", label="Train")
plt.axvline(x=best_tree_size, linewidth=0.5, color="gray", label=best)
plt.title("Decision Tree Classifier Unnormalized Dataset (DTCU)")
plt.ylabel("Mean Average Error")
plt.xlabel("Tree Depth")
plt.legend(bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0.)
plt.show()

dtcu = DecisionTreeClassifier(max_leaf_nodes=best_tree_size, random_state=0)
dtcu.fit(train_images,train_labels)
s_dtcu = dtr.score(test_images,test_labels)
m_dtcu = mean_absolute_error(test_labels, mae.predict(test_images))
print("DTCU Score:", s_dtcu, "\tDTCU MAE:", m_dtcu)
print("DTR Score:", s_dtr, "\tDTR MAE:", m_dtr)
print("DTC Score:", s_dtc, "\tDTC MAE:", m_dtc)
print("SVM Score:", max(n_scores))