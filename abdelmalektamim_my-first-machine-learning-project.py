import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error 

from sklearn.metrics import r2_score

from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_squared_log_error

from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
df=pd.read_excel('../input/number-of-sold-cars-by-country-by-year/cars_sold.xlsx')

print("First rows")

df.head()

print("Last rows")

df.tail()
#define training and testing sets

features=['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']

X=df[features]

X.head()

y=df['2019']

y.head()

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.20, random_state=4)
#scoring model by mean_abs_error for different max_leaf_nodes

mae=[]

for i in range(98):

    model = DecisionTreeRegressor(max_leaf_nodes=i+2,random_state=1)

    model.fit(X_train,y_train)

    y_predict=model.predict(X_test)

    mae.append(mean_absolute_error(y_test, y_predict))

#plotting MAE variation by Max leaf nodes

plt.figure(figsize=(8,8),edgecolor='red')

plt.plot(mae)

plt.ylabel('MAE')

plt.xlabel('Max Leaf Nodes')
#scoring model by r2_score for different max_leaf_nodes

R2=[]

for i in range(98):

    model = DecisionTreeRegressor(max_leaf_nodes=i+2,random_state=1)

    model.fit(X_train,y_train)

    y_predict=model.predict(X_test)

    R2.append(r2_score(y_test, y_predict))

#plotting R2 variation by Max leaf nodes

plt.figure(figsize=(10,10),edgecolor='red')

plt.plot(R2)

plt.ylabel('R2')

plt.xlabel('Max Leaf Nodes')
#scoring model by explained variance for different max_leaf_nodes

Exp_var=[]

for i in range(98):

    model = DecisionTreeRegressor(max_leaf_nodes=i+2,random_state=1)

    model.fit(X_train,y_train)

    y_predict=model.predict(X_test)

    Exp_var.append(explained_variance_score(y_test, y_predict))

#plotting explained variance score variation by Max leaf nodes

plt.figure(figsize=(10,10),edgecolor='red')

plt.plot(Exp_var)

plt.ylabel('Exp_var')

plt.xlabel('Max Leaf Nodes')
#scoring model by mean squarred log for different max_leaf_nodes

m_s_l=[]

for i in range(98):

    model = DecisionTreeRegressor(max_leaf_nodes=i+2,random_state=1)

    model.fit(X_train,y_train)

    y_predict=model.predict(X_test)

    m_s_l.append(mean_squared_log_error(y_test, y_predict))

#plotting mean suarred log variation by Max leaf nodes

plt.figure(figsize=(10,10),edgecolor='red')

plt.plot(m_s_l)

plt.ylabel('mean_squ_log')

plt.xlabel('Max Leaf Nodes')
#define training and testing sets

features=['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']

X=df[features]

X.head()

y=df['2019']

y.head()

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.05, random_state=4)
#scoring model by mean_abs_error for different max_leaf_nodes

mae=[]

for i in range(98):

    model = DecisionTreeRegressor(max_leaf_nodes=i+2,random_state=1)

    model.fit(X_train,y_train)

    y_predict=model.predict(X_test)

    mae.append(mean_absolute_error(y_test, y_predict))

#plotting MAE variation by Max leaf nodes

plt.figure(figsize=(5,5))

plt.plot(mae,color='red')

plt.ylabel('MAE')

plt.xlabel('Max Leaf Nodes')

#scoring model by r2_score for different max_leaf_nodes

R2=[]

for i in range(98):

    model = DecisionTreeRegressor(max_leaf_nodes=i+2,random_state=1)

    model.fit(X_train,y_train)

    y_predict=model.predict(X_test)

    R2.append(r2_score(y_test, y_predict))

#plotting R2 variation by Max leaf nodes

plt.figure(figsize=(5,5))

plt.plot(R2,color='green')

plt.ylabel('R2')

plt.xlabel('Max Leaf Nodes')

#scoring model by explained variance for different max_leaf_nodes

Exp_var=[]

for i in range(98):

    model = DecisionTreeRegressor(max_leaf_nodes=i+2,random_state=1)

    model.fit(X_train,y_train)

    y_predict=model.predict(X_test)

    Exp_var.append(explained_variance_score(y_test, y_predict))

#plotting explained variance score variation by Max leaf nodes

plt.figure(figsize=(5,5))

plt.plot(Exp_var,color='blue')

plt.ylabel('Exp_var')

plt.xlabel('Max Leaf Nodes')

#scoring model by mean squarred log for different max_leaf_nodes

m_s_l=[]

for i in range(98):

    model = DecisionTreeRegressor(max_leaf_nodes=i+2,random_state=1)

    model.fit(X_train,y_train)

    y_predict=model.predict(X_test)

    m_s_l.append(mean_squared_log_error(y_test, y_predict))

#plotting mean suarred log variation by Max leaf nodes

plt.figure(figsize=(5,5),edgecolor='red')

plt.plot(m_s_l,color='magenta')

plt.ylabel('mean_squ_log')

plt.xlabel('Max Leaf Nodes')