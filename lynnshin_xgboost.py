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
train=pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

test=pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
pd.options.display.max_columns = 999

train
# 한번에 전처리

# merge data for preprocessing

alldata = pd.concat([train,test])

alldata
# drop y

alldata2 = alldata.drop(["SalePrice"],1)

alldata2
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

#test

alldata2["Neighborhood"]=le.fit_transform(alldata2["Neighborhood"])

le.classes_ #변환값
alldata2
alldata2.dtypes == object
obj_col = alldata2.columns[alldata2.dtypes == object]

obj_col
alldata2['Alley']
list(alldata2['Alley'])
#label encoding을 할때 all data로 train 과 test를 합쳐서 전처리 해줘야함

#train 과 test를 각각 label encoding으로 했을 때 test에 새로운 데이터가 있을 수도 있음, label이 겹치는 현상

#when processing label encoding, we need to make all data that merged train & test sets as test set may have new data(label) and the label can be overlapped

for i in obj_col:

    alldata2[i]=le.fit_transform(list(alldata2[i]))



alldata2
#결측치 존재 -> nan을 float으로 인식

#결측치 처리 list(alldata2[i])

alldata2=alldata2.fillna(0)
train2 = alldata2[:len(train)]

train2
test2 = alldata2[len(train):]

test2
# #esc + shift + m

# from sklearn.ensemble import RandomForestRegressor

# rf = RandomForestRegressor(n_jobs = -1)

# rf.fit(train2,train["SalePrice"])

# result = rf.predict(test2)
from xgboost import XGBRegressor

xgb = XGBRegressor(learning_rate=0.1)

xgb.fit(train2, train["SalePrice"])

result = xgb.predict(test2)

result
sub = pd.read_csv("/kaggle/input/home-data-for-ml-course/sample_submission.csv")



sub
sub["SalePrice"] = result



sub
sub.to_csv("ens.csv",index=0)