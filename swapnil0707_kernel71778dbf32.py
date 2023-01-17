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
df = pd.read_csv("/kaggle/input/pulsar_stars.csv")

x = df.iloc[:, :8].values

y = df.iloc[:, 8]



# Data Preprocessing



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = 0, strategy = 'mean')

x[:, :8] = imputer.fit_transform(x[:, :8])



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



# Using LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 2)

x_train = lda.fit_transform(x_train, y_train)

x_test = lda.transform(x_test)



# Clasification



from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0, C = 1000, gamma = 0.1)

classifier = classifier.fit(x_train, y_train)



y_pred = classifier.predict(x_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



from sklearn.model_selection import cross_val_score

cvs = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 15)

cvs2 = cross_val_score(estimator = classifier, X = x_test, y = y_test, cv = 15)



l = [cvs.mean(), cvs2.mean()] # 1st element = mean of acc. rate on the training set; 2nd element = On the test set
l