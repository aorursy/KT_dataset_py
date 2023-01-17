#Step 1

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#step 2

import pandas as pd
import numpy as np

df1=pd.read_csv("/kaggle/input/ipl-data/IPL2013.csv")

#Read what features exists in the dataset
cols=df1.columns.tolist()
print(cols)


#Step 3

df1.iloc[:5,:10]
#Step 4


#Step 5

df1.drop(['Sl.NO.','TEAM','AUCTION YEAR'],axis=1)

data=pd.DataFrame(df1)






data=data.drop(['PLAYER NAME','PLAYING ROLE','Sl.NO.','TEAM','AUCTION YEAR','COUNTRY'],axis=1)
data.head()
#Step 6(a)
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]



vif
#Step 6(b)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))

sns.heatmap(vif, annot=True, cmap=plt.cm.Reds)
plt.show()
