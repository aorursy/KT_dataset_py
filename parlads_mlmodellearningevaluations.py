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
# (repeated some of the imports)
from sklearn import (datasets, 
                     metrics, 
                     model_selection as skms,
                     naive_bayes, 
                     neighbors)

# data
iris = datasets.load_iris()

# train-test split
(iris_train_ftrs, iris_test_ftrs, 
 iris_train_tgt, iris_test_tgt) = skms.train_test_split(iris.data,
                                                        iris.target, 
                                                        test_size=.25,
                                                        random_state=42) 
# define some models
models = {'3-NN': neighbors.KNeighborsClassifier(n_neighbors=3),
          '5-NN': neighbors.KNeighborsClassifier(n_neighbors=5),
          'NB'  : naive_bayes.GaussianNB()}

# in turn, fit-predict with those models
for name, model in models.items():
    fit = model.fit(iris_train_ftrs, 
                    iris_train_tgt)
    predictions = fit.predict(iris_test_ftrs)
    
    score = metrics.accuracy_score(iris_test_tgt, predictions)
    print("{:>4s}: {:0.2f}".format(name,score))
%timeit -r datasets.load_iris() 
%%timeit -r1 -n1
(iris_train_ftrs, iris_test_ftrs, 
 iris_train_tgt,  iris_test_tgt) = skms.train_test_split(iris.data,
                                                         iris.target, 
                                                         test_size=.25)
%%timeit -r1

nb    = naive_bayes.GaussianNB()
fit   = nb.fit(iris_train_ftrs, iris_train_tgt)
preds = fit.predict(iris_test_ftrs)

metrics.accuracy_score(iris_test_tgt, preds)
%%timeit -r1

knn    = neighbors.KNeighborsClassifier(n_neighbors=5)
fit   = knn.fit(iris_train_ftrs, iris_train_tgt)
preds = fit.predict(iris_test_ftrs)

metrics.accuracy_score(iris_test_tgt, preds)
%%timeit -r1

knn    = neighbors.KNeighborsClassifier(n_neighbors=3)
fit   = knn.fit(iris_train_ftrs, iris_train_tgt)
preds = fit.predict(iris_test_ftrs)

metrics.accuracy_score(iris_test_tgt, preds)
%load_ext memory_profiler
%%memit 
nb  = naive_bayes.GaussianNB()
fit = nb.fit(iris_train_ftrs, iris_train_tgt)
preds = fit.predict(iris_test_ftrs)
%%memit 
knn = neighbors.KNeighborsClassifier()
fit = knn.fit(iris_train_ftrs, iris_train_tgt)
preds = knn.predict(iris_test_ftrs)