import numpy as np # linear algebra

import pandas as pd # data processing



# plotting

import seaborn as sns

sns.set_style('whitegrid')



# scikit-learn

from sklearn import preprocessing

from sklearn import svm

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import model_selection

from sklearn.decomposition import PCA



import xgboost as xgb



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.concat([pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')])
df.plot(x="LotArea", y="SalePrice", kind="scatter")