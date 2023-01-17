# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#sklearn imports
from sklearn import (datasets, metrics, model_selection as skms, naive_bayes, neighbors)

#warning off
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
iris  = datasets.load_iris()
print(iris.DESCR)
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['tgt'] = iris.target
iris_df.head()
sns.pairplot(iris_df, hue='tgt', height=1.5)
iris.target_names
actual  = np.array([True, True, False, True])
preds = np.array([True, True, True ,True])
correct = actual == preds
correct.sum() / len(actual)
#accuracy score

metrics.accuracy_score(actual, preds)
ftrs = np.random.randint(1,7, (8,2))
ftrs
tgts = np.random.randint(0,2,8)
tgts
plt.scatter(ftrs[:,0] , ftrs[:,1] , c=tgts)
tst = (2,3)
plt.scatter(*tst, c='r')
distance = np.hypot(2-ftrs[:, 0], 3-ftrs[:,1])
distance.argsort()
(iris_train_ftrs, iris_test_ftrs, 
 iris_train_tgt, iris_test_tgt) = skms.train_test_split(iris.data, iris.target, test_size = .70)

len(iris_train_ftrs), len(iris_test_ftrs)
#fitting KNN
knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
fit = knn.fit(iris_train_ftrs, iris_train_tgt)
preds = fit.predict(iris_test_ftrs)

#how accurate it is
actuals = iris_test_tgt
metrics.accuracy_score(actuals,preds)