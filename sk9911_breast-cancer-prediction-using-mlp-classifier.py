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
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.diagnosis.value_counts()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPClassifier



X = df.drop(['id', 'diagnosis', 'Unnamed: 32',], axis=1)

y = df['diagnosis']



scaler = MinMaxScaler()



X_train, X_test, y_train, y_test = train_test_split(X,y)



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)



clf = MLPClassifier(hidden_layer_sizes = [100,100], alpha = 5, solver = 'lbfgs', max_iter = 1000).fit(X_train_scaled, y_train)



print('Training score:', clf.score(X_train_scaled, y_train))

print('Testing score:', clf.score(X_test_scaled, y_test))
from sklearn.model_selection import cross_val_score



acc = cross_val_score(clf, X, y, cv=5);



for score in acc:

    print(f'Accuracy = {score}')