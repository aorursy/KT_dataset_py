# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir('../input'))
label_names = ["No","Yes"]
df_train = pd.read_csv("../input/volcanoes_train/train_images.csv",header=None)
df_test = pd.read_csv("../input/volcanoes_test/test_images.csv",header=None)
print("Shapes training: ",df_train.shape)
print("Shapes test: ",df_test.shape)
df_test.head()
train_labels = pd.read_csv("../input/volcanoes_train/train_labels.csv")
test_labels = pd.read_csv("../input/volcanoes_test/test_labels.csv")
train_labels.head()
sns.countplot(data = train_labels,x="Volcano?")
plt.show()
print("On the ones with volcanoes")
sns.countplot(data = train_labels,x="Type")
plt.show()
sns.countplot(data = train_labels,x="Number Volcanoes")
plt.show()
#Reshape 
X_test = df_test.values.reshape((df_test.shape[0],1,110,110)) 
X_train = df_train.values.reshape((df_train.shape[0],1,110,110))
print(X_train.shape)
#preprocess
X_test = X_test/255.0
X_train = X_train/255.0
#Transpose to tensorflow dimension.
X_test = X_test.transpose([0,2, 3, 1])
X_train = X_train.transpose([0,2, 3, 1])
print(X_train.shape)
train_labels["Volcano?"]
def visualize(X,Y):
    n = np.random.randint(0,X.shape[0])
    aux = X[n]
    
    f,ax = plt.subplots(1,figsize=(8,3))
    ax.set_title("Ground Truth of Volcano?: %s "%(label_names[Y["Volcano?"][n]]))

    ax.imshow(aux[:,:,0],cmap='copper') #the one channel
    ax.set_yticks([])
    ax.set_xticks([])
    plt.show()
    print("Detail:",Y.loc[n,:])
    
visualize(X_train,train_labels)
visualize(X_train,train_labels)
visualize(X_train,train_labels)