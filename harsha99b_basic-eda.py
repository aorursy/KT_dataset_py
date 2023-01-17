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
#Loading the dataset
df = pd.read_csv("/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv")
df.head()
df.describe(include='all')

df.info()
import matplotlib.pyplot as plt
%matplotlib inline

df.hist(figsize=(12,16))
plt.show()

#Length of the whisker = 1.5 * Inter Quartile Range(IQR) 
#IQR = Q3-Q1
import seaborn as sns
sns.boxplot(x="Gender", y="Age", data=df)
plt.show()
pd.crosstab(df['Product'],df['Gender'] )
sns.countplot(x="Product", hue="Gender", data=df)
plt.show()
pd.crosstab(df['Product'],df['Income'] )
pd.pivot_table(df, index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'], aggfunc=len)
sns.pairplot(df)
plt.show()
corr = df.corr(method='pearson')
corr
sns.heatmap(corr, annot=True)
plt.show()
