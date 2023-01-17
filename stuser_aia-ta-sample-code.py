# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/aia-homework-iris-dataset"))
print(os.listdir("../input/test-data"))
# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family']='SimHei' #顯示中文

%matplotlib inline
# Load in the train datasets
train = pd.read_csv('../input/aia-homework-iris-dataset/train.csv', encoding = "utf-8", dtype = {'type': np.int32})
test = pd.read_csv('../input/test-data/test.csv', encoding = "utf-8")
#submission = pd.read_csv('../input/input-iris/submission.csv', encoding = "utf-8", dtype = {'type': np.int32})
train.head(3)
test.head(3)