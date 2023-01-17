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
df = pd.read_csv('/kaggle/input/seeds-dataset/seeds_dataset.txt', sep= '\t', header= None,

                names=['area','perimeter','compactness','lengthOfKernel','widthOfKernel','asymmetryCoefficient',

                      'lengthOfKernelGroove','seedType'])

df.head()
df.shape
df.info()
df.isnull().sum()
df['seedType'].value_counts()
df1 = df.sample(frac = 1, random_state= 3)
df1.head()
X= df1.drop('seedType', axis = 1)

y= df1['seedType']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, random_state=3)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

import lightgbm as lgb
lgbc = lgb.LGBMClassifier()



lgbc = lgb.LGBMClassifier(random_state=3)



params = {'n_estimators': randint(5,100),



'max_depth': randint(2, 20),



'min_child_samples': randint(1, 20),



'num_leaves': randint(5,50)}



rand_search_lgbc = RandomizedSearchCV(lgbc,

param_distributions=params, cv=3, random_state=3, n_jobs=-1)



rand_search_lgbc.fit(X_train, y_train)
print(rand_search_lgbc.best_params_)
predictions = rand_search_lgbc.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confusion_matrix(y_test,predictions)
accuracy_score(y_test,predictions)
print(classification_report(y_test,predictions))