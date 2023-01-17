import warnings

warnings.filterwarnings('ignore')
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

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv('/kaggle/input/penbased-recognition-of-handwritten-digits/penbased-5an-nn.csv')

data.head()
data.info()
X = data.iloc[:,:-1]

X.head()
X.plot.box(figsize=(20,10),xticks=[])

plt.title('Boxplots of all frequency bins')

plt.xlabel('Frequency bin')

plt.ylabel('Power spectral density (normalized)')
X.describe()
y = data.iloc[:,-1]

y.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score,KFold,train_test_split

from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
list_k = []

list_acc = []

for k_value in range(1,10):

    list_k.append(k_value)

    model = KNeighborsClassifier(n_neighbors=k_value)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)*100

    list_acc.append(acc)

    print('accuracy is: ',acc,'for k_value:',k_value)

vitri = list_acc.index(max(list_acc))

k = list_k[vitri]

print('')

plt.plot(list_k,list_acc)

plt.xlabel('number of neighbor k')

plt.ylabel('test accuracy')
models = [

    GaussianNB(),

    KNeighborsClassifier(n_neighbors=3),

    DecisionTreeClassifier(),

    SVC(),

]
CV = 10

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

i=0

for model in models:

    model_name = model.__class__.__name__

    accuracies = cross_val_score(model, X_train, y_train, cv=CV) 

    entries.append([model_name, accuracies.mean()])

    i += 1

cv_df = pd.DataFrame(entries, columns=['model_name', 'accuracy'])
cv_df
model_knn = KNeighborsClassifier(n_neighbors=3)

model_knn.fit(X_train,y_train)

y_pred_knn = model_knn.predict(X_test)

print('accuracy:',accuracy_score(y_test,y_pred_knn))

print("training score:",model_knn.score(X_train,y_train))

print("test score:",model_knn.score(X_test,y_test))
confusion_matrix(y_test,y_pred_knn)
sns.heatmap(confusion_matrix(y_test,y_pred_knn))
print(classification_report(y_test,y_pred_knn))
clf=DecisionTreeClassifier()

clf.fit(X_train,y_train)

y_pred_1=clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred_1))

print("training score:",clf.score(X_train,y_train))

print("test score:",clf.score(X_test,y_test))
confusion_matrix(y_test,y_pred_1)
sns.heatmap(confusion_matrix(y_test,y_pred_1))
print(classification_report(y_test,y_pred_1))
from sklearn.decomposition import PCA
pca = PCA(.95)
pca.fit(X_train)
pca.n_components_
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
list_k = []

list_acc = []

for k_value in range(1,10):

    list_k.append(k_value)

    model = KNeighborsClassifier(n_neighbors=k_value)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)*100

    list_acc.append(acc)

    print('accuracy is: ',acc,'for k_value:',k_value)

vitri = list_acc.index(max(list_acc))

k = list_k[vitri]

print('')

plt.plot(list_k,list_acc)

plt.xlabel('number of neighbor k')

plt.ylabel('test accuracy')
model_knn = KNeighborsClassifier(n_neighbors=3)

model_knn.fit(X_train,y_train)

y_pred_knn = model_knn.predict(X_test)

print('accuracy:',accuracy_score(y_test,y_pred_knn))

print("training score:",model_knn.score(X_train,y_train))

print("test score:",model_knn.score(X_test,y_test))