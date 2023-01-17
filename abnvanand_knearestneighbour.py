import numpy as np

import pandas as pd

from datetime import datetime, date

import os

import matplotlib.pyplot as plt



print(os.listdir("../input"))
bought_df = pd.read_csv('../input/final-bought-dataset.csv')

bought_df.head()
len(bought_df), len(bought_df[bought_df['purchased']==1]), len(bought_df[bought_df['purchased']==0])
df = pd.DataFrame(bought_df)
from sklearn.model_selection import train_test_split



X = df.drop(['purchased'], axis=1)

y = df[['purchased']]

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.30)

X_test, X_validate, y_test, y_validate  = train_test_split(X_test_val, y_test_val, test_size=0.50)

X_train.shape, X_validate.shape, X_test.shape, y_train.shape, y_validate.shape, y_test.shape
## KNN

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=11)

model.fit(X_train, y_train.values.ravel())



y_pred = model.predict(X_validate)
from sklearn.metrics import roc_curve, auc, accuracy_score

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_validate, y_score=y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)



print("Validation set metrics:-")

print("roc_auc", roc_auc)

print("accuracy_score", accuracy_score(y_pred, y_validate))
y_pred = model.predict(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_test.values.ravel(), y_score=y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)



print("Test data metrics:-")

plt.title("ROC curve")

plt.xlabel("false_positive_rate")

plt.ylabel("true_positive_rate")

plt.plot(false_positive_rate,true_positive_rate)



plt.show() 

print("roc_auc", roc_auc)

print("accuracy_score", accuracy_score(y_pred, y_test))
y_pred = pd.DataFrame(y_pred, index=y_test.index, columns=['purchased'])

what_to_buy_df = pd.merge(X_test, y_pred, left_index=True, right_index=True)

what_to_buy_df.head(3)
unique_sid = what_to_buy_df['session'].unique()



fp = open("solution.dat","w")



for sid in unique_sid:

#     all_items = what_to_buy_df['session'==sid , 'item']

    bought_items = what_to_buy_df.loc[(what_to_buy_df['session']==sid) & (what_to_buy_df['purchased']==1), 'item'].values

    if len(bought_items)>0:

        items = ','.join(map(str, bought_items))

        print(f"{sid};{items}") 

        

        print(f"{sid};{','.join(map(str, bought_items))}", file=fp) 



        

fp.close()

# A sample session id with what to buy and not buy

what_to_buy_df[what_to_buy_df['session']==11]