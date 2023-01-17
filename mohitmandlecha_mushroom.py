# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
mush=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
mush
for feature in mush.columns:

    uniq = np.unique(mush[feature])

    print('{}: {} distinct values -  {}'.format(feature,len(uniq),uniq))
mush.describe()
X=mush.drop('class',axis=1)

y=mush['class']

from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

for col in X.columns:

    X[col]=label.fit_transform(X[col])

label1=LabelEncoder()

y=label1.fit_transform(y)
X=pd.get_dummies(X,columns=X.columns,drop_first=True)
X.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
'''from sklearn.decomposition import PCA

pca = PCA(n_components=2)



X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)'''
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression()

classifier.fit(X_train,y_train)
from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
def print_score(classifier,X_train,y_train,X_test,y_test,train=True):

    if train == True:

        print("Training results:\n")

        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train,classifier.predict(X_train))))

        print('Classification Report:\n{}\n'.format(classification_report(y_train,classifier.predict(X_train))))

        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_train,classifier.predict(X_train))))

        res = cross_val_score(classifier, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')

        print('Average Accuracy:\t{0:.4f}\n'.format(res.mean()))

        print('Standard Deviation:\t{0:.4f}'.format(res.std()))

    elif train == False:

        print("Test results:\n")

        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_test,classifier.predict(X_test))))

        print('Classification Report:\n{}\n'.format(classification_report(y_test,classifier.predict(X_test))))

        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_test,classifier.predict(X_test))))

print_score(classifier,X_train,y_train,X_test,y_test,train=True)

print_score(classifier,X_train,y_train,X_test,y_test,train=False)

from sklearn.svm import SVC

classifier = SVC(kernel='rbf',random_state=42)



classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)

print_score(classifier,X_train,y_train,X_test,y_test,train=False)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42)

classifier.fit(X_train, y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)

print_score(classifier,X_train,y_train,X_test,y_test,train=False)

from sklearn.neighbors import KNeighborsClassifier as KNN



classifier = KNN()

classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)

print_score(classifier,X_train,y_train,X_test,y_test,train=False)
