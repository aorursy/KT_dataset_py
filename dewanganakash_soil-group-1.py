# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
pd.set_option('display.max_row',10000)
pd.set_option('display.max_columns',150)

app_train = pd.read_csv('../input/application_train.csv')

plt.figure(figsize=(9,8))
sns.countplot(y=app_train.OCCUPATION_TYPE.values,data=app_train,hue=app_train.TARGET)
plt.xticks(rotation=90)
plt.show()