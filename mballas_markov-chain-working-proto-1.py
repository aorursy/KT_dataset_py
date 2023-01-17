# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import tensorflow as tf 
from tensorflow import keras
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
d=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
d.head()
df = pd.DataFrame(d)
train_data, train_label = df['title'].iloc[0:4500], df['description'].iloc[0:4500]
test_data, test_label =  df['title'].iloc[4501:6233], df['description'].iloc[4501:6233]

train_data, train_label = np.array(train_data), np.array(train_label)
test_data, test_label =  np.array(test_data), np.array(test_label)

print(test_data, test_label)
