# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from PIL import Image

%matplotlib inline

import numpy as np

img=np.array(Image.open('../input/dectree-1/3_class_a.PNG'))

fig=plt.figure(figsize=(10,10))

plt.imshow(img,interpolation='bilinear')

plt.axis('off')

plt.show()
img=np.array(Image.open('../input/dectree-1/Gini_gain_b.PNG'))

fig=plt.figure(figsize=(10,10))

plt.imshow(img,interpolation='bilinear')

plt.axis('off')

plt.show()
img=np.array(Image.open('../input/dectree/Gini_gain_0.5_c.PNG'))

fig=plt.figure(figsize=(10,10))

plt.imshow(img,interpolation='bilinear')

plt.axis('off')

plt.show()
img=np.array(Image.open('../input/dectree/Gini_gain_0.5_d.PNG'))

fig=plt.figure(figsize=(10,10))

plt.imshow(img,interpolation='bilinear')

plt.axis('off')

plt.show()
import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

import warnings

warnings.filterwarnings('ignore') 
dataset=pd.read_csv('../input/socialnetwork/Social_Network.csv')

dataset.head()
X=dataset.iloc[:,[2,3]].values

y=dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split   #cross_validation doesnt work any more

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 

#X_train
from sklearn.preprocessing import StandardScaler 

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.fit_transform(X_test)

#X_train
from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

cm=confusion_matrix(y_test,y_pred)

cm
print(classification_report(y_test,y_pred))
from matplotlib.colors import ListedColormap

X_set,y_set=X_train,y_train

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),

                 np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),

            alpha=0.75,cmap=ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())

plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('Decision Tree (Training set)')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()
from matplotlib.colors import ListedColormap

X_set,y_set=X_test,y_test

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),

                 np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),

            alpha=0.75,cmap=ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())

plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('Decision Tree (Test set)')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()
from sklearn.tree import _tree

def find_rules(tree,features):

    dt = tree.tree_

    def visitor(node,depth):

        indent = ' ' * depth

        if dt.feature[node]!=_tree.TREE_UNDEFINED:

            print('()if <{}> <= {}:'.format(indent,features[node],round(dt.threshold[node],2)))

            visitor(dt.children_left[node],depth+1)

            print('{}else:'.format(indent))

            visitor(dt.children_right[node],depth + 1)

        else:

            print('{}return {}'.format(indent,dt.value[node]))

    visitor(0,1)
f=['Age','EstimatedSalary']
#find_rules(classifier,f)