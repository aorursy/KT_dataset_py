# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
subs = pd.read_csv('../input/sample_submission.csv')
print("Train data shape: ",train.shape)

print("Test data shape: ",test.shape)
train.head()
train.isnull().sum()
sns.distplot(train['sales'],kde = False)

plt.show()
all_data = train.append(test,sort = False)

all_data.shape
all_data.tail()
all_data['date'] = pd.to_datetime(all_data['date'])

all_data.dtypes
all_data['year'] = all_data['date'].dt.year

all_data['month'] = all_data['date'].dt.month

all_data['day'] = all_data['date'].dt.day

all_data['week'] = all_data['date'].dt.week

all_data['weekofyear'] = all_data['date'].dt.weekofyear

all_data['dayofweek'] = all_data['date'].dt.dayofweek

all_data['weekday'] = all_data['date'].dt.weekday

all_data['dayofyear'] = all_data['date'].dt.dayofyear

all_data['quarter'] = all_data['date'].dt.quarter

all_data['is_month_start'] = all_data['date'].dt.is_month_start

all_data['is_month_end'] = all_data['date'].dt.is_month_end

all_data['is_quarter_start'] = all_data['date'].dt.is_quarter_start

all_data['is_quarter_end'] = all_data['date'].dt.is_quarter_end

all_data['is_year_start'] = all_data['date'].dt.is_year_start

all_data['is_year_end'] = all_data['date'].dt.is_year_end
all_data.head()
all_data = all_data.drop('id',axis=1)
categorical_var = ['is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 

                  'is_year_start', 'is_year_end']



for var in categorical_var:

    all_data[var] = all_data[var].astype('category')

    

all_data[categorical_var] = all_data[categorical_var].apply(lambda x: x.cat.codes)
sales = all_data['sales']

all_data = all_data.drop('sales',axis=1)

all_data['sales'] = sales
all_data = all_data.drop('date',axis=1)

all_columns = list(all_data.columns)
corr = all_data[all_columns].corr()

sns.heatmap(corr)
print (corr['sales'].sort_values(ascending=False), '\n') 
all_columns = ['store','item','year','month','day','week','weekofyear','dayofweek','weekday','dayofyear',

               'quarter','is_month_start','is_month_end','is_quarter_start','is_quarter_end','is_year_start',

               'is_year_end']



selected_columns = ['store','item','year','month','day','week','weekofyear','dayofweek','weekday','dayofyear',

               'quarter']



train_new = all_data[all_data['sales'].notnull()]

test_new = all_data[all_data['sales'].isnull()]





X = train_new[selected_columns]

y = train_new['sales']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
model1 = LinearRegression()

model1.fit(X_train, y_train)

predict1 = model1.predict(X_test)
out_df_nn = pd.DataFrame({'id': subs.id.astype(np.int32), 'sales': nn_preds.astype(np.int32)})

out_df_nn.to_csv('submission_nn.csv', index=False)
test_data = test_new[selected_columns] 

test_predict = model1.predict(test_data)
final_out = pd.DataFrame({'id': subs.id, 'sales': test_predict.astype(np.int32)})

final_out.to_csv('submission_reg.csv', index=False)