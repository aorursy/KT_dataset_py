import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn import neighbors

from sklearn import tree

from sklearn import linear_model

import graphviz

import matplotlib.pyplot as plt

import pickle



df = pd.read_csv('../input/dataset2.csv',index_col='country')

df.head()
df = df.drop('Norway',axis=0)

X = df.drop('max_ev_p',axis=1)

y = df['max_ev_p']



train_X, test_X, train_y, test_y = train_test_split(X,y,random_state=0)
new_X = pd.DataFrame(columns=X.columns)

new_X = new_X.append(pd.Series([0,0,0,0,0],index=X.columns,name='zero'))

new_X = new_X.append(pd.Series([0,0,0,0,100],index=X.columns,name='worst'))

new_X = new_X.append(pd.Series([1,0,100000,100,0],index=X.columns,name='best'))

new_X = new_X.append(pd.Series([1,0.56,40000,67,66],index=X.columns,name='country1'))

new_X = new_X.append(pd.Series([0,0.28,55000,78,34],index=X.columns,name='country2'))



new_X
def add_prediction(df,pred):

    df2 = df.copy()

    df2['pred'] = pred

    return df2
for n in [1,2,3,4,5,10,15]:

    knn = neighbors.KNeighborsRegressor(n_neighbors=n)

    knn.fit(train_X,train_y)

    pred = knn.predict(test_X)

    print(mean_squared_error(test_y, pred))
knn = neighbors.KNeighborsRegressor(n_neighbors=2)

knn.fit(train_X,train_y)

pred = knn.predict(test_X)

print(mean_squared_error(test_y, pred))

add_prediction(df.loc[test_X.index], pred)
add_prediction(new_X, knn.predict(new_X))
for n in [1,2,3,4,5,10]:

    dtree = tree.DecisionTreeRegressor(random_state=0,max_depth=n)

    dtree.fit(train_X,train_y)

    dtree_pred = dtree.predict(test_X)

    print(n, mean_squared_error(test_y,dtree_pred))
dtree = tree.DecisionTreeRegressor(random_state=0,max_depth=2)

dtree.fit(train_X,train_y)

dtree_pred = dtree.predict(test_X)

    

add_prediction(df.loc[test_X.index], dtree_pred)
plt.figure(figsize=(3,2))

graphviz.Source(tree.export_graphviz(dtree, feature_names=X.columns))

#graphviz.Source(tree.plot_tree(dtree))
add_prediction(new_X, dtree.predict(new_X))
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(random_state=0,n_estimators=100)

rf.fit(train_X,train_y)

rf_pred = rf.predict(test_X)



print(mean_squared_error(rf_pred, test_y))



add_prediction(df.loc[test_X.index],rf_pred)
reg = linear_model.LinearRegression()



columns = ['ssm','ppp','nat_p']



train_X_reg = train_X[columns].copy()

test_X_reg = test_X[columns].copy()

new_X_reg = new_X[columns].copy()



reg.fit(train_X_reg,train_y)

reg_pred = reg.predict(test_X_reg)



print("MSE:", mean_squared_error(test_y,reg_pred))



print(reg.coef_,train_X_reg.columns)



add_prediction(df.loc[test_X.index], reg.predict(test_X_reg))
add_prediction(new_X_reg, reg.predict(new_X_reg))