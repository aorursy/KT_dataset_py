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
# indexcol=0 は、最初の列である上の要素をインデックスとする(id,host_name,priceとか)

train_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/train.csv', index_col=0)

train_df
test_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/test.csv', index_col=0)

test_df
#dropでは、axis=0で横、1で縦方向に削除

#ちなみにconcatでは、axis=0で縦、1で横方向に連結

df = pd.concat([train_df.drop('price', axis=1), test_df])

df
#dummyはaxis=1で横指定?

#リスト['列名']で指定可能



#これは、列の要素に何があるか知るのに便利！(最終行の後、要素ごとにあれば1、なければ0の2次元配列状に返される)

###df = pd.concat([df, pd.get_dummies(df['neighbourhood_group'])], axis=1)



#要素を数字に置き換える

df['neighbourhood_group'] = df['neighbourhood_group'].map({'Bronx': 0, 'Brooklyn': 1, 'Manhattan': 2, 'Queens': 3, 'Staten Island': 4})



###df = df.drop('neighbourhood_group', axis=1)    元々の配列を消す

df
###df = pd.concat([df, pd.get_dummies(df['neighbourhood'])], axis=1)

df = df.drop('neighbourhood', axis=1)

df
###df = pd.concat([df, pd.get_dummies(df['room_type'])], axis=1)



df['room_type'] = df['room_type'].map({'Private room': 0, 'Entire home/apt': 1, 'Shared room': 2})

###df = df.drop('room_type', axis=1)

df
#Selecting features
df = df.drop(['name', 'host_id', 'host_name', 'last_review', 'reviews_per_month'], axis=1)

df
nrow, ncol = train_df.shape

price_df = train_df[['price']]

train_df = df[:nrow]

train_df = pd.concat([train_df, price_df], axis=1)

train_df
nrow, ncol = train_df.shape

test_df = df[nrow:]

test_df
from sklearn.linear_model import LinearRegression



X = train_df.drop(['price'], axis=1).to_numpy()

y = train_df['price'].to_numpy()



model = LinearRegression()

model.fit(X, y)
X = test_df.to_numpy()



p = model.predict(X)

p = p.astype(int)
submit_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/sampleSubmission.csv')

submit_df['price'] = p

submit_df.to_csv('submission.csv', index=False)