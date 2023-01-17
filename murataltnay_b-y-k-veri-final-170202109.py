import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics 

from sklearn.metrics import r2_score

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

sales_train=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

test=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

submission=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

sales_train.head()
test.head()
submission.head(5)
plt.figure(figsize=(12,6))

sns.scatterplot(sales_train.item_cnt_day,sales_train.item_price)
sales_train= sales_train[sales_train.item_cnt_day<700]

sales_train= sales_train[sales_train.item_price<60000]

plt.figure(figsize=(12,6))

sns.scatterplot(sales_train.item_cnt_day,sales_train.item_price)
sales_train1 = sales_train.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].agg('sum').reset_index()

x = sales_train1.iloc[:,[2,3]]

y = sales_train1.iloc[:,-1:]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

as1=sales_train1[sales_train1['shop_id']==5].reset_index()

as1[as1['item_id']==5037]
model = KNeighborsClassifier(n_neighbors=3) 

model.fit(X_train, y_train) 

y_pred= model.predict(X_test) 

print("R2 Score:",r2_score(y_test,y_pred)) 

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 

print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("R2 Score:",r2_score(y_test,y_pred))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))
rf = RandomForestRegressor(n_estimators=20,random_state=16)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

print("R2 Score:",r2_score(y_test,y_pred))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))
df_test=test.iloc[:,1:]

df_test
rf.fit(x,y)

test_pred=rf.predict(df_test)

pred_df=pd.DataFrame(test_pred,columns=["item_cnt_month",])
pred=pd.DataFrame()

pred['ID']=test['ID']

pred['item_cnt_month']=pred_df['item_cnt_month']/pred_df['item_cnt_month'].max()

submission=pred

submission.head()
submission.to_csv('submission.csv', index=False)