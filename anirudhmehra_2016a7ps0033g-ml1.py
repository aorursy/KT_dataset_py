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
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

# import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/bits-f464-l1/train.csv")

df['t0'] = df['time'] % 1439

df['p0'] = df['time'] // 1439

df.drop('id', axis = 1, inplace = True)

df['label at p-t minus 4'] = df['label'].shift(4*7*1439)

# df['label mean 10 at p-t minus 4'] = df['label'].shift(4*7*1439).rolling(window = 10).mean()

# df['label mean 30 at p-t minus 4'] = df['label'].shift(4*7*1439).rolling(window = 30).mean()

df['around peak?'] = df['t0'].apply(lambda x: 1 if np.abs(x-200) <= 30 else 0)
train_df = df[df['p0'] < 16].dropna()

# test_df = df[df['p0'] > 13]
print(train_df.shape[0] / 1439 / 7)

# print(test_df.shape[0] / 1439 / 7)
X_train = train_df.drop('label', axis = 1)

y_train = train_df['label']

# X_test = test_df.drop('label', axis = 1)

# y_test = test_df['label']
clf = RandomForestRegressor(n_estimators = 250, random_state = 42, n_jobs = -1).fit(X_train, y_train)
# print(mean_squared_error(y_test, clf.predict(X_test)))
# pd.Series(y_test.values - clf.predict(X_test)).plot()

# plt.show()
df = pd.concat([pd.read_csv("/kaggle/input/bits-f464-l1/train.csv"), pd.read_csv("/kaggle/input/bits-f464-l1/test.csv")], sort = False)

df['t0'] = df['time'] % 1439

df['p0'] = df['time'] // 1439

df['label at p-t minus 4'] = df['label'].shift(4*7*1439)

df['around peak?'] = df['t0'].apply(lambda x: 1 if np.abs(x-200) <= 30 else 0)

# df['label mean 10 at p-t minus 4'] = df['label'].shift(4*7*1439).rolling(window = 10).mean()

# df['label mean 30 at p-t minus 4'] = df['label'].shift(4*7*1439).rolling(window = 30).mean()

df = df[df['p0'] > 15]
test_id = df['id']

df.drop(['id', 'label'], axis = 1, inplace = True)
# pd.Series(clf.predict(df)).plot()

# plt.show()
res = pd.DataFrame(data = {'id': test_id, 'label': clf.predict(df)})
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(res)