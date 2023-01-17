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
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report

import xgboost

import lightgbm
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head(5)
df['diagnosis'].value_counts() 
def plot_feature_importance(model, X_train, figsize=(12, 6)):

    sns.set_style('darkgrid')

    

    # Plot feature importance

    feature_importance = model.feature_importances_

    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = np.argsort(feature_importance)

    pos = np.arange(sorted_idx.shape[0]) + .5



    plt.figure(figsize=figsize)

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, X_train.columns[sorted_idx])

    plt.xlabel('Relative Importance')

    plt.title('Variable Importance')

    plt.show()
!pip install dataprep
from dataprep.eda import plot, plot_correlation, plot_missing
y_binary = {'B' : 0 , 'M' : 1}

df['diagnosis'] = df['diagnosis'].map(y_binary)
df.describe()
# 결측값 > 없음

plot_missing(df)
#전체 변수 분포

plot(df)
# target 변수와 다른변수간의 상관관계 확인

plot_correlation(df, "diagnosis") 
# 변수끼리 상관관계

plot_correlation(df)
# target에 상관이 가장 높았던 변수만 regression

plot_correlation(df, x="concave points_worst", y="diagnosis", k=5)