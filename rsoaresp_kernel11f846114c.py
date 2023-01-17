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
from sklearn.metrics import confusion_matrix



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



from sklearn.multiclass import OneVsRestClassifier
data = pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/train.csv')

X, y = data.drop(labels = ['Id', 'Cover_Type'], axis = 'columns'), data['Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier = OneVsRestClassifier(GradientBoostingClassifier(n_estimators = 500))

classifier.fit(X_train, y_train)
test = pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/test.csv')

id_rows, test = test['Id'], test.drop(labels = ['Id'], axis = 'columns')
res = pd.concat([id_rows, pd.DataFrame(classifier.predict(test), columns = ['Cover_Type'])], axis = 'columns')

res.to_csv('samplesubmission.csv', index = False)
print('oi')