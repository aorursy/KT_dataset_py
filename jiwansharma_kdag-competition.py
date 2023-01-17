# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

dataset = pd.read_csv('../input/train_final.csv')

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset.head(5)

for c in ('AREA','SALE_COND','PARK_FACIL','BUILDTYPE','UTILITY_AVAIL','STREET','MZZONE'):

    dataset_c = pd.get_dummies(dataset[c])

    dataset = dataset.drop(c, axis=1)

    dataset = dataset.join(dataset_c)
dataset.head(5)
dataset = dataset.drop('Ana Nagar',axis=1)

dataset = dataset.drop('Ab Normal',axis=1)

dataset = dataset.drop('Adj Land',axis=1)

dataset = dataset.drop('Partiall',axis=1)

dataset = dataset.drop('Noo',axis=1)

dataset = dataset.drop('Comercial',axis=1)

dataset = dataset.drop('Other',axis=1)

dataset = dataset.drop('All Pub',axis=1)

dataset = dataset.drop('NoAccess',axis=1)

dataset = dataset.drop('Pavd',axis=1)

dataset = dataset.drop('Chormpet',axis=1)

dataset = dataset.drop('Chrmpet',axis=1)

dataset = dataset.drop('Chrompt',axis=1)

dataset = dataset.drop('KKNagar',axis=1)

dataset = dataset.drop('Karapakam',axis=1)

dataset = dataset.drop('TNagar',axis=1)

dataset = dataset.drop('Velchery',axis=1)

dataset = dataset.drop('Ann Nagar', axis =1)

dataset.head(5)
y = dataset['SALES_PRICE']



X = dataset.drop('SALES_PRICE',axis=1)

X = X.drop('PRT_ID', axis=1)

da = []

for i in range(0,5331):

        k = X.iloc[i,1]

        da.append(k[6:])



date_sel = pd.DataFrame(da,columns=['DAT_SEL'])

X = X.join(date_sel)

X = X.drop('DATE_SALE',axis =1)

db = []

for i in range(0,5331):

        k = X.iloc[i,5]

        db.append(k[6:])

        

date_build = pd.DataFrame(db,columns = ['DAT_BUILD'])



X = X.join(date_build)

X = X.drop('DATE_BUILD',axis =1)


dc = []

for i in range(0,5331):

        k = int(X.iloc[i,-2])-int(X.iloc[i,-1])

        dc.append(k)

        

buil = pd.DataFrame(dc,columns = ['YEARS'])

X = X.join(buil)
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X)

X = imputer.transform(X)
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

clt1 = XGBRegressor(n_estimators=1000, min_child_weight=1)

clt1.fit(X, y, verbose=False)
clt1.score(X,y)
data = pd.read_csv('../input/test_final.csv')



for c in ('AREA','SALE_COND','PARK_FACIL','BUILDTYPE','UTILITY_AVAIL','STREET','MZZONE'):

    

    dataset_c = pd.get_dummies(data[c])

    data = data.drop(c, axis=1)

    data = data.join(dataset_c)

data.head(10)
data = data.drop('Adyr',axis=1)

data = data.drop('Ab Normal',axis=1)

data = data.drop('Adj Land',axis=1)

data = data.drop('Partiall',axis=1)

data = data.drop('PartiaLl',axis=1)

data = data.drop('Comercial',axis=1)

data = data.drop('Other',axis=1)

data = data.drop('Pavd',axis=1)

data = data.drop('NoAccess',axis=1)

data = data.drop('Chormpet',axis=1)

data = data.drop('Chrmpet',axis=1)

data = data.drop('Chrompt',axis=1)

data = data.drop('Karapakam',axis=1)

data = data.drop('TNagar',axis=1)

data = data.drop('Velchery',axis=1)



x = data.drop('PRT_ID', axis=1)



da = []

for i in range(0,1778):

        k = x.iloc[i,1]

        da.append(k[6:])



date_sel = pd.DataFrame(da,columns = ['DAT_SEL'])

x = x.join(date_sel)

x = x.drop('DATE_SALE',axis =1)

da = []

for i in range(0,1778):

        k = x.iloc[i,5]

        da.append(k[6:])

        

        

date_build = pd.DataFrame(da,columns = ['DAT_BUILD'])

x = x.join(date_build)

x = x.drop('DATE_BUILD',axis =1)



dc = []

for i in range(0,1778):

        k = int(x.iloc[i,-2])-int(x.iloc[i,-1])

        dc.append(k)

        

buil = pd.DataFrame(dc,columns = ['YEARS'])

x = x.join(buil)



from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(x)

x = imputer.transform(x)
y_pred1 = clt1.predict(x)
y_pred1
dx = pd.DataFrame(y_pred1)

dx = dx.join(data['PRT_ID'])

dx.to_csv("predict.csv")
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

create_download_link(dx)