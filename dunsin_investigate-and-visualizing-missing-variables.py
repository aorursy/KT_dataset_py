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
import missingno as msno
import matplotlib.pyplot as plt
df_Train = pd.read_csv('../input/lish-moa/train_features.csv')
df_Test = pd.read_csv('../input/lish-moa/test_features.csv')
df_Train_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
df_Train_unscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
df_Train.isnull().sum()
msno.matrix(df_Train)
msno.matrix(df_Test)
msno.bar(df_Train)
msno.heatmap(df_Train)
msno.heatmap(df_Test)
null_counts = df_Train.isnull().sum()/len(df_Train)
plt.figure(figsize=(16,8))
plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical')
plt.ylabel('fraction of rows with missing data')
plt.bar(np.arange(len(null_counts)),null_counts)
