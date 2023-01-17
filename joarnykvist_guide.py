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
train = pd.read_csv('/kaggle/input/lendo-machine-learning-workshop/train.csv')

train.describe()
import warnings  

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

%matplotlib inline



i = 0

for c in train.columns:

    if c != 'Outcome':

        plt.figure(i)

        plt.xlabel(c)

        train[train['Outcome']==0][c].plot.hist(alpha = 0.5, label='0')

        train[train['Outcome']==1][c].plot.hist(alpha = 0.5, label='1')

        i+=1

        plt.legend(loc='upper right', shadow=True, fontsize='x-large')

        plt.show()

corrMatrix = train.corr()

import seaborn as sn

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(22, 22))

sn.heatmap(corrMatrix, annot=True)

plt.show()
y = train['Outcome']

x = train[[a for a in train.columns if a is not 'Outcome']]



features = [

    'fractal_dimension_se', 

    'symmetry_se',

]



for c in x.columns:

    if c not in features:

        x = x.drop(c, axis=1)



x.head()
from sklearn.linear_model import RidgeClassifier



# Some other models

#from sklearn.neural_network import MLPClassifier 

#from sklearn.ensemble import RandomForestClassifier 

#from sklearn.svm import SVC



from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score



#clf = RandomForestClassifier(n_estimators=10)

#clf = MLPClassifier(hidden_layer_sizes=(2, ))

#clf = SVC()

clf = RidgeClassifier()



clf.fit(x,y)



scores = cross_val_score(clf, x, y, cv=3)

np.mean(scores)
test = pd.read_csv('/kaggle/input/lendo-machine-learning-workshop/test.csv')



for c in test.columns:

    if c not in features:

        test = test.drop(c, axis=1)



predictions = clf.predict(test)



submission = pd.DataFrame(predictions, columns=['Outcome'])

submission['Id'] = submission.index

submission.head()

submission.to_csv('submission.csv', index=False)
