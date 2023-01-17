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
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df= pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

lm=LinearRegression()
X=df['degree_p'].values.reshape(-1,1)
Y=df['mba_p'].values.reshape(-1,1)
lm.fit(X,Y)
yhat=lm.predict(X)
yhat[0:4]
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x='degree_p',y='mba_p',data=df)
lm.score(X,Y)
