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

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from mlxtend.plotting import plot_decision_regions
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv') 
set(df.species)
df.columns
df.head()
sns.pairplot(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
dict ={

    'Iris-setosa': 1 ,

    'Iris-versicolor' : 2,

    'Iris-virginica' : 3

}

def flip(x):

    return dict[x]
df['species'] = df['species'].apply(lambda x : flip(x))
X = df[['sepal_length','sepal_width','petal_length','petal_width']]

y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Using Logistic Regression  

log = LogisticRegression()

log.fit(X_train , y_train)

pred = log.predict(X_test)
pred
score = log.score(X_test, y_test)

print(score)

from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
df.shape
from sklearn.svm import SVC

#svc = SVC(kernel = 'linear')

svc = SVC(kernel = 'poly')

#svc = SVC(kernel = 'sigmoid')

#svc = SVC(kernel = 'rbf')



svc.fit(X_train , y_train)

y_pred = svc.predict(X_test)





from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))







from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pc = pca.fit_transform(X_train)

ldf = pd.DataFrame(data = pc, columns = ['p1', 'p2'])
from sklearn.svm import SVC

#svc = SVC(kernel = 'linear')

#svc = SVC(kernel = 'poly')

#svc = SVC(kernel = 'sigmoid')

svc = SVC(kernel = 'rbf')

svc.fit(ldf, y_train)

pc = pca.fit_transform(X_test)

ldf = pd.DataFrame(data = pc, columns = ['p1', 'p2'])

y_pred = svc.predict(pc)
plot_decision_regions(ldf.values, y_pred,clf =  svc,legend = 1)

# Adding axes annotations

plt.xlabel('sepal length [cm]')

plt.ylabel('petal length [cm]')

plt.title('SVM on Iris')

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv') 
X = df[['sepal_length','sepal_width','petal_length','petal_width']]

Y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)

def create_baseline():

    # create model

    model = Sequential()

    model.add(Dense(4, input_dim=4, activation='relu'))

    model.add(Dense(2, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# evaluate model with standardized dataset

estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, X, encoded_Y, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))