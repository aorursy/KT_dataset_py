import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
#Hot encode the labels to create y_train 
raw = pd.read_csv("../input/train.csv")
labels = raw['Target'].str.split()
labels = np.asarray(labels.values.tolist())
y_train = np.zeros([len(raw['Target']),28]).astype(np.int)
for i in range(len(labels)):
    for j in range (len(labels[i])):
        k = int(labels[i][j])
        y_train[i][k] = 1
#Which images have the same label(multi-label class)
B = np.zeros([len(y_train),len(y_train)]).astype(np.int)
for i in range (len(y_train)):
    A = abs(y_train[i]-y_train)  # square matrix where a row of zeros means the image labels match
    B[i,:] = np.sum(A,axis=1)  #square matrix of scalars.  A zero value means an image label match.  Each row is for a single image compared against all images

#for each row (an image), how many images exist with the same multi-label
HMC = np.array([]).astype(np.int)  # a vector of how many counts for each image
for i in range(len(B)):
    HMC = np.append(HMC,((31072-np.count_nonzero(B[i,:]))-1))# for each row, count the number of nonzero values.  Each zero is a duplicate multi-label class
Unique = np.unique(HMC,return_index=True,return_counts=True)

#now count how many unique classes there are for a given copy count
UCFAGCC = np.zeros_like(Unique[2])    #Unique Class multi-labels For A Given Copy Count
for i in range(len(Unique[2])):
    if i == 0:
        UCFAGCC[i] = Unique[2][i]
    else:
        UCFAGCC[i] = Unique[2][i]/(Unique[0][i] +1)

#now plot out the answer
plt.plot(Unique[0],UCFAGCC,'b.')
plt.xlim(0,120)
plt.xlabel('Count of How Many Copies with that Unique Class Label')
plt.ylabel('Number of Unique Class Labels')
plt. show()
print ("Some Examples")
print ("There are", Unique[2][0], "images in the dataset that have", Unique[0][0],"class copies and therefore",UCFAGCC[0],"unique class labels that are only found",Unique[0][0]+1,"time in X_train")
print ("There are", Unique[2][6], "images in the dataset that have", Unique[0][6],"class copies and therefore",UCFAGCC[6],"unique class labels that are only found",Unique[0][6]+1,"times in X_train")
print ("There are", Unique[2][10], "images in the dataset that have", Unique[0][10], "class copies and therefore",UCFAGCC[10],"unique class labels that are only found",Unique[0][10]+1,"times in X_train")  
print ("There are", Unique[2][30], "images in the dataset that have", Unique[0][30], "class copies and therefore",UCFAGCC[30],"unique class labels that are only found",Unique[0][30]+1,"times in X_train")
print ("There are", Unique[2][60], "images in the dataset that have", Unique[0][60], "class copies and therefore",UCFAGCC[60],"unique class labels that are only found",Unique[0][60]+1,"times in X_train")

