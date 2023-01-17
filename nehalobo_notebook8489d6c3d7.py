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
data=pd.read_csv('/kaggle/input/loan-data-set/loan_data_set.csv')
data

data.mean()
data.median()
data.mode()
numeric = list(data._get_numeric_data().columns)
numeric
categorical = list(set(data.columns) - set(data._get_numeric_data().columns))
categorical
#visualizing on the basis of education and applicantincome
#import matplitlib.pyplot as plt
res = data.groupby(['Education','ApplicantIncome']).size().unstack()
res.plot(kind='bar',figsize=(15,10))
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
correlation=data.corr()
