import pandas as pd
import numpy as np
import math
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from skimage import io
from skimage import feature

import matplotlib.pyplot as plt

import cv2

#LOAD IMAGE DATASET
recurve_path = '/kaggle/input/bow-image/bow/1. traditional Recurve Bow'
longbow_path = '/kaggle/input/bow-image/bow/2. Longbow'
compound_path = '/kaggle/input/bow-image/bow/3. Compound Bow'
crossbow_path = '/kaggle/input/bow-image/bow/4. Crossbow'
#kyudo_path = '/kaggle/input/bow-image/bow/5. Kyudo Bow'

recurve = os.listdir(recurve_path)
longbow = os.listdir(longbow_path)
compound = os.listdir(compound_path)
crossbow = os.listdir(crossbow_path)
#kyudo = os.listdir(kyudo_path)

print('Done')
#VISUALISASI
plt.figure(figsize = (12,12))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = cv2.imread(recurve_path + "/" + recurve[i])
    img = cv2.GaussianBlur(img,(5,5),0)
    plt.imshow(img,cmap='gray')
    plt.title('actual')
    plt.tight_layout()

plt.figure(figsize = (12,12))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = cv2.imread(recurve_path + "/" + recurve[i])
    edges = cv2.Canny(img,25,255,L2gradient=False)
    plt.imshow(edges,cmap='gray')
    plt.title('Seg. canny')
    plt.tight_layout()
    
plt.figure(figsize = (12,12))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = cv2.imread(longbow_path + "/" + longbow[i])
    plt.imshow(img)
    plt.title('actual')
    plt.tight_layout()

plt.figure(figsize = (12,12))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = cv2.imread(longbow_path + "/" + longbow[i])
    edges = cv2.Canny(img,25,255,L2gradient=False)
    plt.imshow(edges,cmap='gray')
    plt.title('Seg. Canny')
    plt.tight_layout()

plt.figure(figsize = (12,12))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = cv2.imread(compound_path + "/" + compound[i])
    plt.imshow(img)
    plt.title('Actual')
    plt.tight_layout()

plt.figure(figsize = (12,12))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = cv2.imread(compound_path + "/" + compound[i])
    edges = cv2.Canny(img,25,255,L2gradient=False)
    plt.imshow(edges,cmap='gray')
    plt.title('Seg. Canny')
    plt.tight_layout()
    
plt.figure(figsize = (12,12))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = cv2.imread(crossbow_path + "/" + crossbow[i])
    plt.imshow(img)
    plt.title('Actual')
    plt.tight_layout()

plt.figure(figsize = (12,12))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = cv2.imread(crossbow_path + "/" + crossbow[i])
    edges = cv2.Canny(img,25,255,L2gradient=False)
    plt.imshow(edges,cmap='gray')
    plt.title('Actual')
    plt.tight_layout()

plt.show()
#plt.figure(figsize = (12,12))
#for i in range(5):
#    plt.subplot(1, 5, i+1)
#    img = cv2.imread(kyudo_path + "/" + kyudo[i])
#    plt.imshow(img)
#    plt.title('Actual')
#    plt.tight_layout()

#plt.figure(figsize = (12,12))
#for i in range(5):
#    plt.subplot(1, 5, i+1)
#    img = cv2.imread(kyudo_path + "/" + kyudo[i])
#    edges = cv2.Canny(img,25,255,L2gradient=False)
#    plt.imshow(edges,cmap='gray')
#    plt.title('Seg. Canny')
#    plt.tight_layout()
#CANNY DAN HUMOMENT
x= 0
x = np.array([['h1','h2','h3','h4','h5','h6','h7','target']])

for i in range(len(recurve)):
    img = cv2.imread('/kaggle/input/bow-image/bow/1. traditional Recurve Bow' + "/" + recurve[i])
    #img = cv2.GaussianBlur(img,(3,3),0)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    #edges = cv2.Sobel(img,cv2.CV_8U,1,1,ksize=5)
    edges = cv2.Canny(img,25,100)
    a = cv2.HuMoments(cv2.moments(edges)).flatten()
    a = np.append(a,1)
    x = np.vstack((x,a))

for i in range(len(longbow)):
    img = cv2.imread('/kaggle/input/bow-image/bow/2. Longbow' + "/" + longbow[i])
    #img = cv2.GaussianBlur(img,(3,3),0)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Sobel(img,cv2.CV_8U,1,1,ksize=5)
    edges = cv2.Canny(img,25,100)
    a = cv2.HuMoments(cv2.moments(edges)).flatten()
    a = np.append(a,2)
    x = np.vstack((x,a))

for i in range(len(compound)):
    img = cv2.imread('/kaggle/input/bow-image/bow/3. Compound Bow' + "/" + compound[i])
    #img = cv2.GaussianBlur(img,(3,3),0)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Sobel(img,cv2.CV_8U,1,1,ksize=5)
    edges = cv2.Canny(img,25,100)
    a = cv2.HuMoments(cv2.moments(edges)).flatten()
    a = np.append(a,3)
    x = np.vstack((x,a))

for i in range(len(crossbow)):
    img = cv2.imread('/kaggle/input/bow-image/bow/4. Crossbow' + "/" + crossbow[i])
    #img = cv2.GaussianBlur(img,(3,3),0)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Sobel(img,cv2.CV_8U,1,1,ksize=5)
    edges = cv2.Canny(img,25,100)
    a = cv2.HuMoments(cv2.moments(edges)).flatten()
    a = np.append(a,4)
    x = np.vstack((x,a))

#for i in range(len(kyudo)):
#    img = cv2.imread(kyudo_path + "/" + kyudo[i])
    #img = cv2.GaussianBlur(img,(3,3),0)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
#    edges = cv2.Canny(img,25,100)
 #   a = cv2.HuMoments(cv2.moments(edges)).flatten()
 #   a = np.append(a,'kyudo')
 #   x = np.vstack((x,a))

print('Done')
#EXPORT to CSV
np.savetxt("/kaggle/working/bowcanny.csv", x, fmt='%s',delimiter=',' )
print('Done')
#LOAD CSV DATASET
dataset = pd.read_csv('/kaggle/working/bowcanny.csv')
print (len(dataset))
print (dataset)
#split target and attribute
x = dataset.iloc[:,1:7]
y = dataset.iloc[:,7]

#split train n test dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0, test_size=0.1)
print(len(y_test))
print(len(x_train))
print(len(dataset))
#SPLITTING VISUALIZATION
plt.figure(figsize=(10,13))
plt.subplot(2,2,1);y_train.value_counts().plot(kind='bar', color=['C0','C1','C2','C3','C4','C5','C6']);plt.title('training')
plt.subplot(2,2,2);y_test.value_counts().plot(kind='bar', color=['C0','C1','C2','C3','C4','C5','C6']);plt.title('testing')
plt.subplot(2,2,3);y_train.value_counts().plot(kind='pie');plt.title('training')
plt.subplot(2,2,4);y_test.value_counts().plot(kind='pie',);plt.title('testing')
#scaling data
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
x_train
#choose method
method =  DecisionTreeClassifier(max_depth=4)

#crossvalidation
accuracy = cross_val_score(method,x,y, cv=5, scoring='accuracy')
precision = cross_val_score(method,x,y, cv=5, scoring='precision_weighted')
recall = cross_val_score(method,x,y, cv=5, scoring='recall_weighted')
f1 = cross_val_score(method,x,y, cv=5, scoring='f1_weighted')
print('accuray',  accuracy)
print('precision' , precision)
print('recall' ,recall)
print('F1-Score' , f1)



#BOXPLOT VISUALIZATION

fig1, ax1 = plt.subplots(figsize=(10,5))

#green_diamond = dict(markerfacecolor='g', marker='D')
red_square = dict(markerfacecolor='r', marker='s')


# grouping
all_data = [accuracy,precision,recall,f1]
ax1.set_title('performance - boxplot')

# plot box plot
ax1.boxplot(all_data,notch=False,flierprops=red_square)




#adding horizontal grid lines
ax1.yaxis.grid(True)
ax1.set_xticks([y +1 for y in range(len(all_data))])
ax1.set_xlabel('performa')
ax1.set_ylabel('score')

#add x-tick labels
plt.setp(ax1, xticks=[y+1 for y in range(len(all_data))],
         xticklabels=[ 'accuracy','precision','recall','f1_score'])
plt.show()
plt.figure(figsize=(10,5))
xx = ["cv1", "cv2", "cv3", "cv4", "cv5"] #, "cv6", "cv7", "cv8", "cv9", "cv10"
plt.plot(xx, accuracy, '^')
plt.plot(xx, precision, '--')
plt.plot(xx, recall, '--')
plt.plot(xx, f1, '--')
plt.title("comparison of each crossvalidation - SVM")
plt.xlabel("Crossvaldiation")
plt.ylabel("score")
plt.legend(["accuracy","precision", "recall", "f1-score"])
plt.grid()
plt.show()