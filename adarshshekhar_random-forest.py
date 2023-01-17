# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
def read_and_fix(fname):
    df=pd.read_csv(fname)
    df['Dates']=pd.to_datetime(df["Dates"])
    df['Year']=df['Dates'].dt.year
    df['Month']=df['Dates'].dt.month
    df['Hour']=df['Dates'].dt.hour
    month_map={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['Month'].replace(month_map, inplace=True)
    df['Year']=df['Year'].astype(str)
    df['Hour']=df['Hour'].astype(str)
    
    return df
train=read_and_fix('../input/train.csv')
print(train.shape)
train.head()
feature_columns=['X','Y','PdDistrict','Hour']
X=train[feature_columns]
print(X.shape)
print(X.dtypes)
from sklearn import preprocessing
le_p=preprocessing.LabelEncoder()
X.loc[:,'PdDistrict']=le_p.fit_transform(X['PdDistrict'])
X.loc[:,'Hour']=pd.to_numeric(X['Hour'])
print(X.dtypes)
print(X.head())
y=train['Category']
le_y=preprocessing.LabelEncoder()
y_num=le_y.fit_transform(y)
print(type(y))

