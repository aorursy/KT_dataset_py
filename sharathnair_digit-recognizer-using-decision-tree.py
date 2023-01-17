# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
#constants

IMG_HEIGHT=28
IMG_WIDTH=28

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
loaded_images=pd.read_csv('../input/train.csv')
loaded_images.head()
images=loaded_images.iloc[:,1:]
labels=loaded_images.iloc[:,:1]   # for the labels to be a dataframe . iloc[:,0] returns a Series  iloc[:,:1] returns a Dataframe
labels.head()
train_images,test_images,train_labels,test_labels=train_test_split(images,labels,test_size=0.2,random_state=13)
train_images.describe()
tree=DecisionTreeClassifier(criterion='gini',random_state=1)
tree.fit(train_images,train_labels)
tree.score(train_images,train_labels.values.ravel())
tree.score(test_images,test_labels.values.ravel())
figr,axes=plt.subplots(figsize=(10,10),ncols=3,nrows=3)
axes=axes.flatten()
for i in range(0,9):
    jj=np.random.randint(0,test_images.shape[0])          #pick a random image
    axes[i].imshow(test_images.iloc[[jj]].values.reshape(IMG_HEIGHT,IMG_WIDTH))
    axes[i].set_title('predicted: '+str(tree.predict(test_images.iloc[[jj]])[0]))



new_data=pd.read_csv('../input/test.csv')
new_data.head(n=3)
y_pred=tree.predict(new_data)
y_pred.shape
submissions=pd.DataFrame({"ImageId":list(range(1,len(y_pred)+1)), "Label":y_pred})
submissions.head()
submissions.to_csv("mnist_decision_tree_submit.csv",index=False,header=True)
!ls
