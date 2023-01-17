import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import CategoricalNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def convert_from_binary(df, column_names, new_feature_name):
  ''' 
  Takes a list of columns (column_names), where every row in the dataframe df
  has exactly one value in these columns set to 1 and the rest set to 0, and
  creates a new column named new_feature_name that has the name of the column
  that is set to 1 for each row as its value. The returned dataframe has this
  new feature and does not have the old columns. 
  E.g., if the columns were ["eviv1", "eviv2", "eviv3"] and new_feature_name was
  floorrating, then if row 3 had a 1 for eviv2 and 0s for eviv1 and eviv3, row 3
  in the returned dataframe would have floorrating set to "eviv2".
  '''
  df_subset = df.loc[:,column_names].idxmax(axis=1)
  df = df.drop(columns=column_names)
  df[new_feature_name] = df_subset
  return df
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

path = '/kaggle/input/matches.csv'
df = pd.read_csv(path)

df['win_by_runs'][df['win_by_runs'] != 0] = 1
df['win_by_wickets'][df['win_by_wickets'] != 0] = 1
win_type = convert_from_binary(df, ['win_by_runs', 'win_by_wickets'], 'win_type')
# df.loc[:,'win_type'] = win_type # Adding this column reduced accuracy by 0.04
df = df.drop(columns=['Season', 'city', 'id', 'date', 'dl_applied', 'result', 'win_by_runs', 'win_by_wickets', 'player_of_match', 'umpire3', 'umpire1', 'umpire2'])
df = df.astype(str)
le = preprocessing.LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['winner'])
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = CategoricalNB(alpha=12)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)