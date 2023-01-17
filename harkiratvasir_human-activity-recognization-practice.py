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
import pandas as pd
dataset = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/train.csv')
dataset.head()
dataset.shape
dataset.isnull().any().sort_values()
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize = (15,10))

sns.countplot(dataset['Activity'])
X = dataset.drop(['Activity','subject'],axis = 1)

y = dataset.subject
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.decomposition import PCA

pca = PCA(n_components=None)

principalComponents = pca.fit_transform(X)

#First i will check pca for all the components then i will check most explainable varaibles
import matplotlib.pyplot as plt

df = pd.DataFrame(pca.explained_variance_ratio_)
print(df.max(),df.min())
df.sort_values(by = 0,ascending = False)
df[0][:50].sum()*100
from sklearn.decomposition import PCA

pca = PCA(n_components=50)

principalComponents = pca.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.decomposition import PCA

pca = PCA(n_components=None)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
df = pd.DataFrame(pca.explained_variance_ratio_)
df[:200].sum()

#50 variable explained 87.65 percent of the variance
from sklearn.decomposition import PCA

pca = PCA(n_components=150)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
accuracy_scores = np.zeros(4)
# Import all machine learning algorithms

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# Import other useful subpackage

from sklearn.metrics import accuracy_score
# Support Vector Classifier

clf = SVC().fit(X_train, y_train)

prediction = clf.predict(X_test)

accuracy_scores[0] = accuracy_score(y_test, prediction)*100

print('Support Vector Classifier accuracy: {}%'.format(accuracy_scores[0]))
# Logistic Regression

clf = LogisticRegression().fit(X_train, y_train)

prediction = clf.predict(X_test)

accuracy_scores[1] = accuracy_score(y_test, prediction)*100

print('Logistic Regression accuracy: {}%'.format(accuracy_scores[1]))
clf = KNeighborsClassifier().fit(X_train, y_train)

prediction = clf.predict(X_test)

accuracy_scores[2] = accuracy_score(y_test, prediction)*100

print('K Nearest Neighbors Classifier accuracy: {}%'.format(accuracy_scores[2]))
# Random Forest

clf = RandomForestClassifier().fit(X_train, y_train)

prediction = clf.predict(X_test)

accuracy_scores[3] = accuracy_score(y_test, prediction)*100

print('Random Forest Classifier accuracy: {}%'.format(accuracy_scores[3]))
labels = ['SVM','LR','KNN','RF']

plt.bar(labels,accuracy_scores)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Logistic Regression gives 96.1 accuracy on human activity dataset



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset1 = pd.read_csv('../input/human-activity-recognition-with-smartphones/train.csv')

X_train= dataset.iloc[:,:-1].values

y_train= dataset.iloc[:, -1].values



dataset2 = pd.read_csv('../input/human-activity-recognition-with-smartphones/test.csv')

X_test= dataset.iloc[:,:-1].values

y_test= dataset.iloc[:, -1].values



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0,max_iter = 1000)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score

cm = confusion_matrix(y_test, y_pred)

accuracy_score=accuracy_score(y_test,y_pred)

recall_score=recall_score(y_test,y_pred,average='weighted')

f1_score=f1_score(y_test,y_pred,average='weighted')

print(y_pred)

print(cm)

print(accuracy_score)

print(recall_score)

print(f1_score)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Logistic Regression gives 96.1 accuracy on human activity dataset



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('../input/human-activity-recognition-with-smartphones/train.csv')

X= dataset.iloc[:,:-1].values

y= dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)





# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0,max_iter = 1000)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score

cm = confusion_matrix(y_test, y_pred)

accuracy_score=accuracy_score(y_test,y_pred)

recall_score=recall_score(y_test,y_pred,average='weighted')

f1_score=f1_score(y_test,y_pred,average='weighted')

print(y_pred)

print(cm)

print(accuracy_score)

print(recall_score)

print(f1_score)