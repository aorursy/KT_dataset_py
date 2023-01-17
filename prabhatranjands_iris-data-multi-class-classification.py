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
cols=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

df= pd.read_csv('/kaggle/input/multiclassification-iris-dataset/iris.csv', names=cols)

df.head() ## display first 5 observation
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
df.shape
print(df.describe())
df['class'].value_counts()
df.plot(kind='box', subplots=True,layout=(2,2))
plt.show()
df.hist()
plt.show()
scatter_matrix(df)
plt.show()
import seaborn as sns
sns.pairplot(df)
array=df.values
x=array[:,0:4]
y=array[:,4]
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=20, random_state=1)
models = []
models.append(('Logistic Regression:-',LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA:-',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART:-', DecisionTreeClassifier()))
models.append(('NB:-',GaussianNB()))
models.append(('SVM:-',SVC(gamma='auto')))

# evaluation of model 
results= []
names= []

for name, model in models:
    kfold=StratifiedKFold(n_splits=10, random_state=1,shuffle=True)
    cv_result= cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_result)
    names.append(name)
    print('(%s: %f %f)' % (names,cv_result.mean(),cv_result.std()))
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
