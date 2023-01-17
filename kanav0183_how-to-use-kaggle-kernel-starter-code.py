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
train = pd.read_csv('../input/train_final.csv')

test = pd.read_csv('../input/test_final.csv')
train.head()
ids = test.PRT_ID
del train['PRT_ID']
del test['PRT_ID']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from tqdm import  tqdm
from sklearn.preprocessing import LabelEncoder
cat_cols = [f for f in train.columns if (train[f].dtype == 'object')]
for col in tqdm(cat_cols):
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))
train.head()
x = train.drop('SALES_PRICE',axis=1)

y=train.SALES_PRICE
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

x.fillna(0,inplace=True)

reg.fit(x,y)



price = reg.predict(test.fillna(0))

sub = pd.DataFrame(ids)
sub['SALES_PRICE'] = price
sub.to_csv('sample1.csv',index=None)
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
create_download_link(sub)
