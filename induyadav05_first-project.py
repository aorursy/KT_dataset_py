# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")
train.head()

test.head()
# Import the libraries 

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#Divide the train set into depenedent and independent variable

train_X = train.iloc[:,[2,4,5,9]].values

train_Y = train.iloc[:,1].values

test_X = test.iloc[:,[1,3,4,8]].values
# Manage the missing values in the dependent variable 

from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(train_X[:,2:4])

train_X[:,2:4] = imputer.transform(train_X[:,2:4])

imputer.fit(test_X[:,2:4])

test_X[:,2:4] = imputer.transform(test_X[:,2:4])

print(train_X)

print(test_X)
# Change the categorical dependent variable into binary form

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_X[:,1] = le.fit_transform(train_X[:,1])

test_X[:,1] = le.fit_transform(test_X[:,1])

print(train_X)

print(test_X)
# feature scaling to scale the each feature into the same range

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

train_X = sc_X.fit_transform(train_X)

test_X = sc_X.transform(test_X)
# Since the test dataset has no dependent variable.so, it is not possible to calculate the 

# confusion matrix to check the accuracy of the model. hence, the feature reduction technique 

# Principal component analysis (PCA) is used so that model can be visualized in 2D to check how model is predicting.

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

train_X = pca.fit_transform(train_X)

test_X = pca.transform(test_X)

explained_variance = pca.explained_variance_ratio_
# Fitting the training set to the Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(train_X,train_Y)
# Predicting the dependent variable from the test set

test_pred = classifier.predict(test_X)

test_Y = np.round(test_pred)
# visualising the Training set results

from matplotlib.colors import ListedColormap

X_set, y_set =train_X,train_Y

X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),

                    np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap =ListedColormap(('red', 'green' )))

plt.xlim(X1.min(), X1.max())

plt.ylim(X1.min(), X1.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c =ListedColormap(('red', 'green'))(i), label = j)

plt.title('Titanic : Survival Prediction by Logistic Regression Model (Training set)')

plt.xlabel('PC-1')

plt.ylabel('PC-2')

plt.legend()

plt.show()
# visualising the Test set results

from matplotlib.colors import ListedColormap

X_set, y_set =test_X,test_Y

X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.1),

                    np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.1))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap =ListedColormap(('red', 'green' )))

plt.xlim(X1.min(), X1.max())

plt.ylim(X1.min(), X1.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c =ListedColormap(('red', 'green'))(i), label = j)

plt.title('Titanic : Survival Prediction by Logistic Regression Model (Test set)')

plt.xlabel('PC-1')

plt.ylabel('PC-2')

plt.legend()

plt.show()
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test_Y})
my_submission.head()
my_submission.to_csv('submission.csv', index=False)