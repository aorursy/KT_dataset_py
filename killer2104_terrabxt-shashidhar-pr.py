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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('../input/into-the-future/train.csv')
df.tail()
df.drop('time',axis=1,inplace=True)
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
x.shape,y.shape
sns.scatterplot('feature_1','feature_2',data=df)
df1=pd.read_csv('../input/into-the-future/test.csv')
df1.head()
df1.drop('time',axis=1,inplace=True)
df1.info()
df.info()
df.shape,df1.shape
corr=df.corr()

sns.heatmap(corr)
sns.boxplot(df.feature_1,data=df)
df.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
from sklearn import metrics
metrics.r2_score(y_test,pred)
rmse=((y_test-pred)**2).sum()
np.sqrt(np.mean((y_test-pred)**2))
y_pred=model.predict(df1)
df1['feature_2']=y_pred
df1.head()
df1.to_excel("output.xlsx") 



