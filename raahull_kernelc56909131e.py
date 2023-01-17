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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
X_train = pd.read_csv("/kaggle/input/titanic/train.csv")
X_test = pd.read_csv("/kaggle/input/titanic/test.csv")
X_train.head()
X_train.drop(columns = ["Name","Ticket","PassengerId","Survived"])
X_test.drop(columns = ["Name","Ticket","PassengerId","Survived"])
numerical_cols = ["Pclass",]
numerical_trans = SimpleImputer(strategy = "constant")
categorical_trans = Pipeline(steps=[('imputer',SimpleImputer(strategy = 'most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num',numerical_trans,numerical_cols),('cat',categorical_trans,categorical_cols)])


