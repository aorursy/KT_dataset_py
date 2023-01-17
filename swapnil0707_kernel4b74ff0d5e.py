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
df = pd.read_csv("/kaggle/input/diabetes.csv")

x = df.iloc[:, 0:8].values

y = df.iloc[:, 8]



# Data Preprocessing



from sklearn.impute import SimpleImputer

imputer1 = SimpleImputer(missing_values = 0, strategy = 'constant', fill_value = 23)

x[:, 3] = imputer1.fit_transform(x[:, 3].reshape(-1, 1)).reshape(768)

imputer3 = SimpleImputer(missing_values = 0, strategy = 'constant', fill_value = 26.5)

x[:, 5] = imputer3.fit_transform(x[:, 5].reshape(-1, 1)).reshape(768)

imputer2 = SimpleImputer(missing_values = 0, strategy = 'mean')

x[:, [1,2, 4, 6, 7]] = imputer2.fit_transform(x[:, [1,2,4,6,7]])



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.17, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



# Using LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 1)

x_train = lda.fit_transform(x_train, y_train)

x_test = lda.transform(x_test)



# Clasification



from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0, C = 300, gamma = 0.09)

classifier = classifier.fit(x_train, y_train)



y_pred = classifier.predict(x_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



from sklearn.model_selection import cross_val_score

cvs = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)



from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)

l = [cvs.mean(), cvs.std(), acc]
l
# Data Preprocessing



from sklearn.impute import SimpleImputer

imputer1 = SimpleImputer(missing_values = 0, strategy = 'constant', fill_value = 23)

x[:, 3] = imputer1.fit_transform(x[:, 3].reshape(-1, 1)).reshape(768)

imputer3 = SimpleImputer(missing_values = 0, strategy = 'constant', fill_value = 26.5)

x[:, 5] = imputer3.fit_transform(x[:, 5].reshape(-1, 1)).reshape(768)

imputer2 = SimpleImputer(missing_values = 0, strategy = 'mean')

x[:, [1,2, 4, 6]] = imputer2.fit_transform(x[:, [1,2,4,6]])



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.17, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



# Using ANN



import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score



def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))

    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)



classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)



from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)

l = [acc, accuracies.mean(), accuracies.std()]

l
# Data Preprocessing



from sklearn.impute import SimpleImputer

imputer1 = SimpleImputer(missing_values = 0, strategy = 'constant', fill_value = 23)

x[:, 3] = imputer1.fit_transform(x[:, 3].reshape(-1, 1)).reshape(768)

imputer3 = SimpleImputer(missing_values = 0, strategy = 'constant', fill_value = 26.5)

x[:, 5] = imputer3.fit_transform(x[:, 5].reshape(-1, 1)).reshape(768)

imputer2 = SimpleImputer(missing_values = 0, strategy = 'mean')

x[:, [1,2, 4, 6]] = imputer2.fit_transform(x[:, [1,2,4,6]])



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.17, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



# Using PCA



from sklearn.decomposition import PCA

lda = PCA(n_components = 7)

x_train = lda.fit_transform(x_train)

x_test = lda.transform(x_test)



# Using ANN



import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score



def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier



classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)



classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)



from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)

l = [acc, accuracies.mean(), accuracies.std()]

l
df = pd.read_csv("/kaggle/input/diabetes.csv")

x = df.iloc[:, 0:8].values

y = df.iloc[:, 8]



# Data Preprocessing



from sklearn.impute import SimpleImputer

imputer1 = SimpleImputer(missing_values = 0, strategy = 'constant', fill_value = 23)

x[:, 3] = imputer1.fit_transform(x[:, 3].reshape(-1, 1)).reshape(768)

imputer3 = SimpleImputer(missing_values = 0, strategy = 'constant', fill_value = 26.5)

x[:, 5] = imputer3.fit_transform(x[:, 5].reshape(-1, 1)).reshape(768)

imputer2 = SimpleImputer(missing_values = 0, strategy = 'mean')

x[:, [1,2, 4, 6, 7]] = imputer2.fit_transform(x[:, [1,2,4,6,7]])



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.17, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



# Fitting XGBoost to the Training set

from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(x_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(x_test)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)

from sklearn.metrics import accuracy_score

l = [accuracies.mean(), accuracies.std(), accuracy_score(y_test, y_pred)]

l