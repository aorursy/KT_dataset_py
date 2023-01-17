# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/sign-language-mnist'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score



df = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

df_test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')



x_train = df.iloc[0:27455, 1:785].values

y_train = df.iloc[0:27455, 0].values



x_test = df_test.iloc[0:7172, 1:785].values

y_test = df_test.iloc[0:7172,0].values



label_enc = LabelEncoder()

y_train = label_enc.fit_transform(y_train)

y_test = label_enc.fit_transform(y_test)



from sklearn.svm import SVC



classifier = SVC(decision_function_shape='ovr')



classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)



acc = accuracy_score(y_test,y_pred)

f1 = f1_score(y_test,y_pred,average='micro')

cm = confusion_matrix(y_test,y_pred)



print(cm)

print(f1)

print(acc)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score



df = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

df_test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')



x_train = df.iloc[0:27455, 1:785].values

y_train = df.iloc[0:27455, 0].values



pixel_number = np.arange(0,784,1)



x_test = df_test.iloc[0:7172, 1:785].values

y_test = df_test.iloc[0:7172,0].values



plt.scatter(x_train[0],pixel_number, s=0.4, c = 'r')

plt.scatter(x_train[1],pixel_number, s=0.4, c = 'b')

plt.scatter(x_train[2],pixel_number, s=0.4, c = 'g')

plt.scatter(x_train[3],pixel_number, s=0.4, c = 'y')

plt.scatter(x_train[4],pixel_number, s=0.4, c = 'm')

plt.show()



label_enc = LabelEncoder()

y_train = label_enc.fit_transform(y_train)

y_test = label_enc.fit_transform(y_test)



from sklearn.neighbors import KNeighborsClassifier



KNN = KNeighborsClassifier(n_neighbors=165)

classifier = KNN.fit(x_train,y_train)



y_pred = classifier.predict(x_test)

acc = accuracy_score(y_test,y_pred)

f1 = f1_score(y_test,y_pred,average='micro')

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(f1)

print(acc)