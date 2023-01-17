# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load the iris dataset

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names = ['sepal length', 'sepal width','petal length', 'petal width', 'class'])

iris.head()
#create train and test set

X = iris.drop('class', axis=1).values

Y = iris['class'].values

X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state=0)
# compute accuracy for 10 folds

lr = LogisticRegression()



kfold = KFold(n_splits=10, shuffle= True, random_state=1)

scores = []



for k, (train, test) in enumerate(kfold.split(X_train, Y_train)):

    

    lr.fit(X_train[train], Y_train[train])

    score = lr.score(X_train[test],Y_train[test])

    scores.append(score)

    print("Fold %d: Accuracy=%.4f" %(k,score))
# compute mean accuracy

accuracy = np.array(scores).mean()

print('Validation accuracy = %.4f' % accuracy)
# alternative: use score validation instead

from sklearn.model_selection import cross_val_score



lr = LogisticRegression()



score = cross_val_score(lr, X_train, Y_train, cv = 20)

score.mean()
# train the model on the whole test test before starting predictions

lr.fit(X_train, Y_train)