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
images = labeled_images.iloc[0:5000,1:] # Mengambil data sebanyak 5000 dan mengambil semua kolom kecuali kolom pertama (yang akan mencjadi image)
labels = labeled_images.iloc[0:5000,:1] # Mengambil data sebanyak 5000 dan mengambil kolom pertama (yang akan menjadi label atau penanda angka)
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)  # membagi data dengan test_size = 0,2 dari data sepenuhnya
# now we gonna load the second image, reshape it as matrix than display it
i=1 # Tentukan urutan array manakah yang ingin diperlihatkan
img=train_images.iloc[i].values # memangil index array dengan memilih bedasarkan posisi integernya lalu dimasukkan kedalam array
img=img.reshape((28,28)) # mebentuk array kedalam bentuk 28x28
plt.imshow(img,cmap='gray') # plot data tersebut oleh img dan beri lah dia warna dengan cmap
plt.title(train_labels.iloc[i,0]) # beri judul plot tersebut
# Todo: Put your code here
# Coba-coba
#print(train_labels)
print(type(train_labels))
print(type(train_labels.iloc[i].label))
# membuat array dengan panjang sesuai dengan label yang kita punya
class_i = [0 for i in range(len(np.unique(test_labels)))]
# print (class_i)
# telah diketahui bahwa array class_i berjumlah 10 yang artinya digit hanya mempunyai digit 0-9
# membuat iterasi sepanjang train_labels untuk mencari index masing masing 0-9
# dan jika sudah mendapat setiap nilai 0-9 sudah mendapat indexnya masing-masing maka iterasi selesai
for i in range(len(train_labels)):
    if (class_i[train_labels.iloc[i].label] != 0):
        continue # jika index label yang diinginkan sudah terisi langsung berlanjut ke iterasi berikutnya
    else:
        class_i[train_labels.iloc[i].label] = i # masukkan index tersebut ke array yang kita buat

print(class_i)
for u in class_i:
    plt.figure(u) # agar nantinya img bisa membuat figure baru jadi yang keluar outputnya akan sesuai tidak hanya 1 gambar
    img = train_images.iloc[u].values
    img = img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[u,0])
# Contoh
train_images.iloc[i].describe()
print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])
plt.title(train_labels.iloc[i,0])
plt.xlabel("Pixel")
plt.ylabel("Banyaknya")
# Semua class disatukan dalam satu histogram
for p in class_i:
    plt.hist(train_images.iloc[p])
    plt.title("0-9")
    plt.xlabel("Pixel")
    plt.ylabel("Banyaknya")
# Semua class dengan masing masing histogramnya
for z in class_i:
    plt.figure(z)
    plt.hist(train_images.iloc[z])
    plt.title(train_labels.iloc[z,0])
    plt.xlabel("Pixel")
    plt.ylabel("Banyaknya")
# create histogram for each class (data merged per class)
# Contoh
# Todo
# print(train_labels.iloc[:10])
# Memargin kedua data
data1 = train_images.iloc[1]
#print(data1)
data2 = train_images.iloc[3]
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.append(data1,data2)
print(type(data3))
#print(len(data3))
plt.hist(data3)
# harus bisa memargin 10 class

data = []
for z in range (10):
    a = class_i[z]
    data1 = train_images.iloc[a].values
    for p in data1 :
        data.append(p)
plt.hist(data)
plt.title("Jumlah Pixel di semua class 0 - 9")
plt.xlabel("Pixel")
plt.ylabel("Banyaknya")
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# Put your verification code here
# Todo
print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
model = DecisionTreeRegressor(random_state=0)
model.fit(train_images, train_labels)
predict = model.predict(test_images)
mean_absolute_error(test_labels, predict)
test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
# now plot again the histogram
plt.hist(train_images.iloc[i])
# Score
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
# MAE
model = DecisionTreeRegressor(random_state=0)
model.fit(train_images, train_labels)
predict = model.predict(test_images)
mean_absolute_error(test_labels, predict)
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
from sklearn.metrics import accuracy_score
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
svm_model = svm.SVC(kernel='linear', C=1, gamma='auto')
svm_model.fit(train_images,train_labels.values.ravel())
predictions = svm_model.predict(test_images)
print(accuracy_score(predictions, test_labels))
print(mean_absolute_error(predictions,test_labels))
from sklearn.tree import DecisionTreeClassifier
DTC = pd.read_csv('../input/train.csv')
images = DTC.iloc[0:5000,1:] # Mengambil data sebanyak 5000 dan mengambil semua kolom kecuali kolom pertama (yang akan mencjadi image)
labels = DTC.iloc[0:5000,:1] # Mengambil data sebanyak 5000 dan mengambil kolom pertama (yang akan menjadi label atau penanda angka)
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
DTC = DecisionTreeClassifier()
DTC.fit(train_images, train_labels)
predict = DTC.predict(test_images)
mean_absolute_error(test_labels, predict)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_images, train_labels)
    preds_val = model.predict(test_images)
    test_mae = mean_absolute_error(test_labels, preds_val)
    return(test_mae)

def get_mae_train(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(test_images, test_labels)
    preds_train = model.predict(train_images)
    train_mae = mean_absolute_error(train_labels, preds_train)
    return(train_mae)

test = []
train = []
candidate_max_leaf_nodes = [5, 25, 50, 75, 100, 125, 150, 175, 200]
for max_leaf_nodes in candidate_max_leaf_nodes:
    mae_test = get_mae(max_leaf_nodes, train_images, test_images,train_labels, test_labels)
    test.append(mae_test)
    mae_train = get_mae_train(max_leaf_nodes, train_images, test_images,train_labels, test_labels)
    train.append(mae_train)

plt.plot(candidate_max_leaf_nodes, test, color = "Blue")
plt.plot(candidate_max_leaf_nodes, train, color = "Red")
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
# value normalized to [0,1]
test_images[test_images>0]=1
train_images[train_images>0]=1
DTC.fit(train_images, train_labels)
predict = DTC.predict(test_images)
mean_absolute_error(test_labels, predict)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_images, train_labels)
    preds_val = model.predict(test_images)
    test_mae = mean_absolute_error(test_labels, preds_val)
    return(test_mae)

def get_mae_train(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(test_images, test_labels)
    preds_train = model.predict(train_images)
    train_mae = mean_absolute_error(train_labels, preds_train)
    return(train_mae)

test = []
train = []
candidate_max_leaf_nodes = [5, 25, 50, 75, 100, 125, 150, 175, 200]
for max_leaf_nodes in candidate_max_leaf_nodes:
    mae_test = get_mae(max_leaf_nodes, train_images, test_images,train_labels, test_labels)
    test.append(mae_test)
    mae_train = get_mae_train(max_leaf_nodes, train_images, test_images,train_labels, test_labels)
    train.append(mae_train)

plt.plot(candidate_max_leaf_nodes, test, color = "Green")
plt.plot(candidate_max_leaf_nodes, train, color = "Red")
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
DTR = pd.read_csv('../input/train.csv')
images = DTR.iloc[0:5000,1:] # Mengambil data sebanyak 5000 dan mengambil semua kolom kecuali kolom pertama (yang akan mencjadi image)
labels = DTR.iloc[0:5000,:1] # Mengambil data sebanyak 5000 dan mengambil kolom pertama (yang akan menjadi label atau penanda angka)
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
DTR = DecisionTreeRegressor()
DTR.fit(train_images, train_labels)
predict = DTR.predict(test_images)
mean_absolute_error(test_labels, predict)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_images, train_labels)
    preds_val = model.predict(test_images)
    test_mae = mean_absolute_error(test_labels, preds_val)
    return(test_mae)

def get_mae_train(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(test_images, test_labels)
    preds_train = model.predict(train_images)
    train_mae = mean_absolute_error(train_labels, preds_train)
    return(train_mae)

test = []
train = []
candidate_max_leaf_nodes = [5, 25, 50, 75, 100, 125, 150, 175, 200]
for max_leaf_nodes in candidate_max_leaf_nodes:
    mae_test = get_mae(max_leaf_nodes, train_images, test_images,train_labels, test_labels)
    test.append(mae_test)
    mae_train = get_mae_train(max_leaf_nodes, train_images, test_images,train_labels, test_labels)
    train.append(mae_train)

plt.plot(candidate_max_leaf_nodes, test, color = "Blue")
plt.plot(candidate_max_leaf_nodes, train, color = "Green")
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
