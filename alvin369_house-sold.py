# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.chdir("../input")
# Any results you write to the current directory are saved as output.
data=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
len(data.columns)
data.dtypes.sample(81)
data.GarageType.unique()
data[['Street','SalePrice']][data.Street=="Pave"]
df=data.select_dtypes(include=['int64','float64'])
df.head()
df_drop=df.drop(columns=['SalePrice','Id'])
df_drop.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import Imputer
imp=Imputer()
trans_data=imp.fit_transform(df_drop)
model=RandomForestRegressor()
model.fit(trans_data,data.SalePrice)
test.head()
transformedTest=test[df_drop.columns]
val=imp.fit_transform(transformedTest)
predicted_value=model.predict(val)
d={'Id':test.Id,'SalePrice':predicted_value}
fin=pd.DataFrame(data=d)
fin.to_csv("output.csv")
os.listdir()
x=[34,5435,53,6546,54]
ty=pd.DataFrame(x)
ty.to_csv("sva.csv")





