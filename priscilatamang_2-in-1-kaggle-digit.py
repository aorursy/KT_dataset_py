# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test.head()
train.isnull().sum()
test.isnull().sum()
print(train.shape)
print(test.shape)
X=train.iloc[:,1:].values
print("The shape of X:",X.shape)

Y=train.iloc[:,0].values
print("The shape of Y:",Y.shape)
#Splitting
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

print("The shape of X_train:",X_train.shape)
print("The shape of Y_train:",Y_train.shape)
print(Y_test.shape)
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
#Predicting
Y_predict=knn.predict(X_test) 
print(Y_predict) 

#Finding Accuracy
AS1=accuracy_score(Y_test,Y_predict) 
print("The accuracy score using knn:", AS1)
np.sqrt(X_train.shape[0])
X_test=test.iloc[:,:].values
Y_test=knn.predict(X_test)
test.index
test.index.tolist()
imageid=test.index.tolist()
#Converting that index starting from 0 to the one starting from 1

ImageId = [x+1 for x in imageid]
ImageId
print(len(ImageId))
print(Y_test.shape)
Final = pd.DataFrame()
Final

Final['ImageId'] = ImageId
Final['Label'] = Y_test
Final
#Converting it into csv file

Final.to_csv('submission.csv', index=False)