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
train = pd.read_csv('../input/train__updated.csv')

test = pd.read_csv('../input/test__updated.csv')
train.head()
train.drop('index',axis=1,inplace=True)

idx=test['index']

test.drop('index',axis=1,inplace=True)

from sklearn.linear_model import*

reg = LinearRegression()



X=train[test.columns]

y=train.wave_height
X.fillna(-1,inplace=True)

test.fillna(-1,inplace=True)
reg=LinearRegression()
reg.fit(X,y)



pred =reg.predict(test)



sam=pd.read_csv('../input/sample_sub.csv')



sam['wave_height']=pred
sam.to_csv('sandbox_sub.csv',index=None)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe

create_download_link(sam)