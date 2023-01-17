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
['IMDB-Movie-Data.csv']
import pandas as pd
mv_rev = pd.read_csv('../input/IMDB-Movie-Data.csv')
mv_rev.describe()
mv_rev.head()
mv_rev.isnull()
mv_rev=mv_rev.replace(0,np.NaN)
mv_rev.head(10)
mv_rev.fillna(mv_rev.mean(),inplace=True)
mv_rev.head(10)
import seaborn as sns
mv_rev.columns=['rank','title','genre','desc','dir','actors','year','runtime','rating','votes','revenue','metascore']
sns.pairplot(mv_rev)
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features=mv_rev[['rating', 'revenue', 'votes','rank','runtime']]
np_scaled = min_max_scaler.fit_transform(features)
df_normalized = pd.DataFrame(np_scaled)

labels=mv_rev['metascore']
print(df_normalized)

print(labels)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df_normalized,labels,test_size=0.3,train_size=0.7,random_state=34)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg_fit=reg.fit(x_train,y_train)
pred=reg_fit.predict(x_test)
from sklearn.metrics import r2_score
score=r2_score(y_test,pred)
print(score)
