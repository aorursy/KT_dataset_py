import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

import os
print(os.listdir('../input/'))

titanic = pd.read_csv('../input/train.csv')
titanic.head()
titanic.dtypes
titanic.isnull().sum()
titanic.info()
titanic.describe()
titanic.corr()
plt.figure(figsize=(18,12))
sns.heatmap(titanic.corr(), annot=True)