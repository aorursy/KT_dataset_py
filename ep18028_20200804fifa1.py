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
train_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv', index_col=0)
train_df.head()
test_df.head()
train_df.info()
test_df.info()
train_df
import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt



sns.set_style("darkgrid")

pyplot.figure(figsize=(40, 40))  # 図の大きさを大き目に設定

sns.heatmap(train_df.corr(), square=True, annot=True)  # 相関係数でヒートマップを作成
aaa = ['overall','potential','skill_moves','attacking_short_passing','skill_long_passing','skill_ball_control','movement_reactions','power_shot_power','mentality_vision','mentality_composure']



X_train = train_df[aaa].to_numpy()

y_train = train_df['value_eur'].to_numpy()

X_test = test_df[aaa].to_numpy()
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier()

model.fit(X_train, y_train)

p_test = model.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)

submit_df['value_eur'] = p_test

submit_df
submit_df.to_csv('submission.csv')