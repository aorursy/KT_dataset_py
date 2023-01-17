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

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import imblearn
df = pd.read_csv('../input/mushroom-csv-file/mushrooms.csv')

df.head()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for col in df.columns:

    df[col] = encoder.fit_transform(df[col])

    

df.head()
X = df.drop('class',axis=1)

y = df['class']
y.value_counts()
from imblearn.under_sampling import NearMiss
nm=NearMiss()

X_res,y_res = nm.fit_sample(X,y)
y_res.value_counts()
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_res= sc.fit_transform(X_res)
from sklearn.decomposition import PCA

pca = PCA(n_components=4)



X_res = pca.fit_transform(X_res)
X_res.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=101)
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors=2)



model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
def print_score(model,X_train,y_train,X_test,y_test,train=True):

    if train == True:

        print("Training results:\n")

        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train,model.predict(X_train))))

        print('Classification Report:\n{}\n'.format(classification_report(y_train,model.predict(X_train))))

        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_train,model.predict(X_train))))

        res = cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')

       

    elif train == False:

        print("Test results:\n")

        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_test,model.predict(X_test))))

        print('Classification Report:\n{}\n'.format(classification_report(y_test,model.predict(X_test))))

        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_test,model.predict(X_test))))
print_score(model,X_train,y_train,X_test,y_test,train=True)
print_score(model,X_train,y_train,X_test,y_test,train=False)