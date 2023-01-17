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
import pandas as pd
import numpy as np
import os
#os.chdir('D:\\Pro')
#os.getcwd()
df= pd.read_csv("/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv")
df.describe()
df.columns
df.isnull().sum()
df.shape
df.describe()
df= df.drop(index=32,axis=0)
print(df.columns)
df.shape
x_train= df
y_train= df['diagnosis']

diagnosis = pd.get_dummies(x_train['diagnosis'],prefix='Diag')
diagnosis.head()
diagnosis= diagnosis.drop(['Diag_M'], axis=1)
diagnosis['Diag_B'].value_counts()
x_train= x_train.drop(['diagnosis'],axis=1)
#x_train=pd.concat([diagnosis, x_train], axis=1)
y_train= diagnosis
y_train.head()
x_train.columns
y_train= diagnosis
y_train.columns
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
#x_train= x_train.drop(['diagnosis'], axis=1)

X_test=x_train
Y_train=y_train


print(x_train.head())
print(y_train.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train,y_train, test_size = 0.2, random_state = 0)
model1 = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
model1.fit(X_train,y_train.values.ravel())
predicted1 = model1.predict(X_test);
print("Logistic Reg",accuracy_score(y_test,predicted1)*100,"%")
model5 = DecisionTreeClassifier(criterion= 'entropy', max_depth=5)
model5.fit(X_train,y_train)
predicted5 = model5.predict(X_test)
print("Decision Tree",accuracy_score(y_test,predicted5)*100,"%")

model5
