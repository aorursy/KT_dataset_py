# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_train.head()
df_test = pd.read_csv('../input/test.csv')
df_test.head()
df_train['family_size']= 1 + df_train.SibSp + df_train.Parch
df_train.head()
df_test['family_size']= 1 + df_test.SibSp + df_test.Parch
df_test.head()
df_train.shape
df_test.shape
df_train.describe()
df_test.describe()
import seaborn as sns

from matplotlib import pyplot as plt
%matplotlib inline