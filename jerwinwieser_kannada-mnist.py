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



from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
df_train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

df_test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

df_train.shape, df_test.shape
label = df_train.pop('label')

id = df_test.pop('id')

df_train.shape, df_test.shape


X_train, X_eval, y_train, y_eval = train_test_split(df_train, label, test_size=.3, random_state=42)

X_train.shape, X_eval.shape, y_train.shape, y_eval.shape
def display(i):

    img = np.reshape(X_train.values[i],[28,28])

    lab = y_train.values[i]

    plt.figure(figsize=(2,2))

    plt.imshow(img,cmap='gray')

    plt.title('label : {}'.format(lab))

    plt.xticks([])

    plt.yticks([])

    plt.show()
nimg = 1

p = [display(i) for i in range(0,nimg)]
# apply model on a evaluation set

# clf = MLPClassifier(solver='adam', alpha=1e-5,

#                     hidden_layer_sizes=(10,), random_state=42)

# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_eval)

# accuracy_score(y_eval, y_pred)
# apply model on a testing set

clf = MLPClassifier(solver='adam', alpha=1e-5,

                    hidden_layer_sizes=(10,), random_state=42)

clf.fit(df_train, label)

y_pred = clf.predict(df_test)
df_submission = pd.DataFrame({'id': id, 'label': y_pred})

df_submission.to_csv('/kaggle/working/submission.csv', index=False)