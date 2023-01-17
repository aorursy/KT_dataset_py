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
df=pd.read_csv('../input/into-the-future/train.csv')
df['time']=pd.to_datetime(df['time'])
df.tail()
df['time']=df['time'].astype('category').cat.codes
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
x= pca.fit_transform(df.iloc[:,1:3].values)

y=df.iloc[:,[3]].values
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000, max_depth=500, n_jobs=-1, random_state=0)
model.fit(x,y)
res=model.predict(x)

res=res.reshape(len(res),1)

df1=pd.read_csv('../input/into-the-future/test.csv')
df1['time']=pd.to_datetime(df1['time'])
df1['time']=df1['time'].astype('category').cat.codes
x1=pca.fit_transform(df1.iloc[:,1:3].values)
res1 =model.predict(x1)

test_predict=res1.reshape(len(res1),1)
os.chdir(r'/kaggle/working')
import numpy as np
import pandas as pd
prediction = pd.DataFrame(test_predict, columns=['feature_2']).to_csv('myprediction.csv',index = None)

prediction
