# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


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
prefs =  """cat tea    red
            cat coffee red
            cat tea    red
            dog coffee red
            cat tea    black
            dog coffee black
            dog coffee black
            dog tea    black""".split()

df = pd.DataFrame(np.array(prefs).reshape(8,3), columns=['pet', 'drink', 'color'])
df
pd.crosstab(df.drink, df.color)
pd.crosstab(df.pet, df.color)
#color

red_data = df.groupby('color').get_group('red')
red_data.pet.value_counts(normalize = True)
#drinks

red_data.drink.value_counts(normalize = True)
#overall berakdown of colors

df.color.value_counts(normalize=True)
#probablity of red vs black -----> when we see the cat , coffee
# So what is the probablity of coler being red when cat , coffee?
# in match ----> p(coffee/red) * p(cat/red) * p(red)
#         ------> .5             .75           .5
# compare the value with black is the case 
#red is more likely result
iris  = datasets.load_iris()
(iris_train_ftrs, iris_test_ftrs, 
 iris_train_tgt, iris_test_tgt) = skms.train_test_split(iris.data, iris.target, test_size = .70)

len(iris_train_ftrs), len(iris_test_ftrs)
nb = naive_bayes.GaussianNB()
fit= nb.fit(iris_train_ftrs, iris_train_tgt)
preds = fit.predict(iris_test_ftrs)


metrics.accuracy_score(iris_test_tgt, preds)