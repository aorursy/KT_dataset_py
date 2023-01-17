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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
warnings.filterwarnings('ignore')

train = pd.read_csv("/kaggle/input/titanic/train.csv")
train["Cabin"].unique()
# 再頻出値を確認
print(train["Cabin"].mode())

# 再頻出値を確認
train1=train.fillna({"Cabin":"G6"})
train1.head(6)

train2=train
mf_cabin = ['B96 B98' , 'C23 C25 C27', 'G6']

#Ageの欠損値をランダムな値で埋める
train2["Cabin"][pd.isnull(train2["Cabin"])] = random.choices(mf_cabin)

train2.head()
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train3 = train


train3["Cabin"][pd.notnull(train3["Cabin"])] = "1"
train3["Cabin"][pd.isnull(train3["Cabin"])] = "0"


train3.head()