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
train_df = pd.read_csv('../input/1056lab-football-players-value-prediction/train.csv', index_col=0)
test_df = pd.read_csv('../input/1056lab-football-players-value-prediction/test.csv', index_col=0)
train_df


numeric_columns = ['age','height_cm','weight_kg','overall','potential']


X_train = train_df[numeric_columns].to_numpy()
y_train = train_df['value_eur'].to_numpy()
X_test = test_df[numeric_columns].to_numpy()


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
p_test = model.predict(X_test)
submit_df = pd.read_csv('../input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)
submit_df['value_eur'] = p_test
submit_df
submit_df.to_csv('submission.csv')