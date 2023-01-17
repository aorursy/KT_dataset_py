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
import pandas as pd

census_income_learn = pd.read_csv("../input/census_income_learn.csv",header=None)

census_income_test = pd.read_csv("../input/census_income_test.csv",header=None)
census_income_learn.describe()
X = census_income_learn



# on supprime les lignes qui ont des valeurs manquantes

X_ss_NA = X.dropna(axis=0, inplace=False)



y = X_ss_NA.iloc[:,-1]



X_ss_NA.shape

# on supprime la colonne y

X_f = X_ss_NA.drop([41], axis=1, inplace=False)



#y.head()
# selection colonnes valeurs numériques

numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

numeric_cols
import numpy as np

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

np.unique(y)
import seaborn as sns

sns.distplot(numeric_cols[0]);