
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv('../input/bill_authentication/bill_authentication.csv')

dataset.head()
X=dataset.iloc[:,[0,1,2,3]].values
y=dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
dataset.hist(figsize=(8,8))
sns.pairplot(dataset)

corr_matrix=dataset.corr()


sns.heatmap(corr_matrix,annot=True)

from sklearn.tree import DecisionTreeRegressor

DecisionTree_model=DecisionTreeRegressor()
DecisionTree_model.fit(X_train,y_train)
accuracy=DecisionTree_model.score(X_test,y_test)
accuracy
from sklearn.ensemble import RandomForestRegressor

RandomForest_model=RandomForestRegressor(n_estimators=200,max_depth=10)
RandomForest_model.fit(X_train,y_train)
accuracy=RandomForest_model.score(X_test,y_test)
accuracy
