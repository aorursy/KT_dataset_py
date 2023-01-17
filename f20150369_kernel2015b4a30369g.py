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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')

y=data['label']

x=data.drop(labels=['label','id'],axis=1)
import numpy as np

import matplotlib.pyplot as plt

from  sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

import sklearn



scaler= MinMaxScaler().fit(x)

scaled_data=scaler.transform(x)

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=42)

regressor= RandomForestRegressor(max_depth=30,n_estimators=100).fit(X_train,y_train)

y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error

print("mse is: ",mean_squared_error(y_test, y_pred),sep='')
test = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')

tids=test['id']

xt=test.drop(labels=['id'],axis=1)
scaled_data_test=scaler.transform(xt)

y_pred= regressor.predict(xt)
ans=pd.DataFrame(data={'id':tids,'label':y_pred})

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

 csv = ans.to_csv(index=False)

 b64 = base64.b64encode(csv.encode())

 payload = b64.decode()

 html = '<a download="sub2.csv" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

 html = html.format(payload=payload,title=title,filename=filename)

 return HTML(html)

create_download_link("sub.csv")