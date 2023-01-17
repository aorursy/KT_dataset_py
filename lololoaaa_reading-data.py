# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/income_train.csv', header=0)
df.shape
df.describe()
data_df_numeric_column_name=list(df.select_dtypes(include=[np.number]))
data_df_numeric = df[data_df_numeric_column_name]
data_df_column_name = list(df)
print(data_df_numeric_column_name)
print()
# find all category columns
data_df_category_column_name = [a for a in data_df_column_name if a not in data_df_numeric_column_name]
data_df_category = df[data_df_category_column_name]
print(data_df_category_column_name)
print()