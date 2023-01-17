# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read input data
#train.csv -> Contains the training data.
#test.csv -> Contains the testing data for which we have to predict the category.

train_data = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/train.csv.zip', compression='zip')
test_data = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/test.csv.zip', compression='zip')
# print the shape of train and test dataframes to get a view of the number of columns and rows.
print("Shape of Testing Data")
print(test_data.shape)
print("Shape of Training Data")
print(train_data.shape)
print("\n")


#training data
train_data.head()
#testing data
test_data.head()
#print the columns present in Training and Testing data.
print("Columns in Training Data")
print(train_data.columns)
print("\n")

print("Columns in Testing Data")
print(test_data.columns)
train_data.describe()
test_data.describe()

#From the above information we colud see that 'bone_length', 'rotting_flesh', 'hair_length' and 'has_soul' are float values. 
#'color' and 'type' have categorical values. Below, we will try to get more about each of the columns.
#also id 
X = pd.get_dummies(train_data.drop(['color','type','id'], axis = 1))
Y= train_data['type']
trainid=train_data.values[:,0]
testid=test_data.values[:,0]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

X=np.array(X_train)
Y=np.array(y_train)
#MLP classifier
model=MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=(5,), random_state=0)
model.fit(X,Y)
print(model)
y_pred=model.predict(X_test)
#Training accuracy
y_pred=model.predict(X_test)
model.score(X_test,y_test)
#Validation accuracy

scores= cross_val_score(model, X, Y,cv=10)
scores
scores.mean()
#final prediction
test=test_data.drop(['color','id'],axis=1)
prediction=model.predict(test)

submission=pd.DataFrame({'id':testid, 'type': prediction})
submission.to_csv("submission.csv",index=False)
