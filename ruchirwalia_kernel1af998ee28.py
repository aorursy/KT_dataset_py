# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
df.head()
df.shape
df= df.values
# df to numpy convert
#list to array
X = df[:,1:]
Y= df[:,0]
def draw(sample):
    img=sample.reshape([28,28])
    plt.imshow(img, cmap="gray")
    plt.show()
draw(X[45])
print(Y[45])
split = int(X.shape[0]*0.9)
X_train = X[:split,:]
Y_train = Y[:split]

X_test = X[:split,:]
Y_test = Y[:split]
print(X_train.shape , Y_train.shape)
print(X_test.shape , Y_test.shape)
def distance(x1 , x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X , Y , query ,  k = 5):
    vals = []
    m = X.shape[0]
    for i in range(m-1):
        dist = distance(X[i] , query)
        vals.append([dist , Y[i]])

    vals = sorted(vals)

    vals = vals[:k]
    vals = np.array(vals)

    #print(vals[:,1])
    new_val = np.unique(vals [:,1] , return_counts = True)

    #print(new_val)

    index = new_val[1].argmax()

    #print(index)

    ans = new_val[0][index]

    #print(new_val[0][index])
    
    return int(ans)

pred= knn(X_train,Y_train,X_test[10],k=21)
print(pred)
print(Y_test[10])
draw(X_test[10])
m=10
for i in range(m):
    pred= knn(X_train,Y_train, X_train[i])
    if pred==Y_test[i]:
        count+=1
    
acc= ((count)/(10))*100
prin