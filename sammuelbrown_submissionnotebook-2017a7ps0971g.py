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
from IPython.display import HTML

import base64



def create_download_link(df, title = "Download CSV File", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    

    html = html.format(payload=payload, title=title, filename=filename)

    return HTML(html)
train_data = pd.read_csv("/kaggle/input/bits-f464-l1/train.csv")

test_data = pd.read_csv("/kaggle/input/bits-f464-l1/test.csv")
train_df = train_data.copy(deep=True)

train_df.drop(["id","label"],axis=1,inplace=True)
test_df = test_data.copy(deep=True)

test_df.drop(["id"],axis=1,inplace=True)
train_y = train_data["label"]
zero_deviation = []
for col in train_df.columns:

    if(train_df[col].std() == 0.0):

        zero_deviation.append(col)
train_df.drop(zero_deviation,axis=1,inplace=True)

test_df.drop(zero_deviation,axis=1,inplace=True)
from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()



train_df[train_df.columns] = scaler.fit_transform(train_df[train_df.columns])

test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])
corr = train_df.corr().abs()
upper_triangle = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))



to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
to_drop
to_drop = to_drop[:-3]
train_df_uncor = train_df.drop(to_drop, axis=1)

test_df_uncor = test_df.drop(to_drop, axis=1)
len(train_df_uncor.columns)
from sklearn.ensemble import RandomForestRegressor


model = RandomForestRegressor(n_estimators = 500, random_state=0)

model.fit(train_df_uncor,train_y)
y_pred = model.predict(test_df_uncor)
final_df = pd.DataFrame()
final_df["id"] = test_data["id"]
final_df["label"] = y_pred
final_df.head()
create_download_link(final_df)
len(final_df)