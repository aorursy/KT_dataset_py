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
df = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20')

df.head(10)
df.isnull().any()
df= df.drop('Timestamp',1)

p = df.loc[df['Weighted_Price'].idxmax()]

print (p)
from sklearn.preprocessing import MinMaxScaler

cols = df.columns.values

print (cols)
Min_max_scaler = MinMaxScaler()

df[cols] = Min_max_scaler.fit_transform(df[cols])

df.head()
X = df.drop('Weighted_Price',1).values

y = df['Weighted_Price']
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X_train,y_train)
print ('Residual sum of squares Train: %.2f' % np.mean((model.predict(X_train)- y_train) ** 2))

print ('Residual sum of squares Test: %.2f' % np.mean((model.predict(X_test)- y_test) ** 2))