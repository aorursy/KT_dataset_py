import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/others/Movie_regression.xls',header=0) #since our csv file has header at 0th row, we use header=o
df.head(10)
df.columns
df.shape
df['Genre'].unique()
df.info()
df['Time_taken'].mean()
df['Time_taken'].fillna(value=df['Time_taken'].mean(),inplace=True) 
#we have filled the missing values with mean values
df.info()
df.head()
# 3d and Genre are categorical 
# we will convert them into dummy variable

df=pd.get_dummies(df,columns=['3D_available','Genre'],drop_first=True) #drop_first = n-1 , 
df.head()
df.shape
X=df.loc[:,df.columns!='Collection']
type(X)
X.head()
X.shape
y=df['Collection']
type(y)
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# since we are randomly assigning our data into test and train , to get the same test data everytime, so that 
#we can compare the performmace of the data
#if i keep random state the same, we will get the same train test split
X_train.head() #indexes are shuffled, 
X_train.shape
X_test.shape
from sklearn import tree
regtree=tree.DecisionTreeRegressor(max_depth=3)
# max depth = no of layers in our tree, we dont want to overfit, we use 3 . 
# Don't exceed beyond 5
regtree.fit(X_train,y_train)
y_train_pred=regtree.predict(X_train)
y_test_pred=regtree.predict(X_test)
y_test_pred
from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error(y_test,y_test_pred)
# here we give test values and predicted values for y
r2_score(y_train,y_train_pred)
# the value obtained is 0.83 which means our model is performing great
# calculate r2 values on our test data
r2_score(y_test,y_test_pred)

#always look at your test r2 values to evaluate your model performance
dot_data=tree.export_graphviz(regtree, out_file=None)
from IPython.display import Image
import pydotplus
graph=pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
regtree1=tree.DecisionTreeRegressor(max_depth=3)
regtree1.fit(X_train,y_train)
dot_data=tree.export_graphviz(regtree1, out_file=None,feature_names=X_train.columns,filled=True) #filled = it will fill colors as per the conditon for the target variable = collection
graph1=pydotplus.graph_from_dot_data(dot_data)
Image(graph1.create_png())

regtree2=tree.DecisionTreeRegressor(min_samples_split=40)
regtree2.fit(X_train,y_train)
dot_data=tree.export_graphviz(regtree2, out_file=None,feature_names=X_train.columns,filled=True) 
graph2=pydotplus.graph_from_dot_data(dot_data)
Image(graph2.create_png())
regtree3=tree.DecisionTreeRegressor(min_samples_leaf=25)
regtree3.fit(X_train,y_train)
dot_data=tree.export_graphviz(regtree3, out_file=None,feature_names=X_train.columns,filled=True) 
graph3=pydotplus.graph_from_dot_data(dot_data)
Image(graph3.create_png())
regtree3=tree.DecisionTreeRegressor(min_samples_leaf=25,max_depth=4)
regtree3.fit(X_train,y_train)
dot_data=tree.export_graphviz(regtree3, out_file=None,feature_names=X_train.columns,filled=True) 
graph3=pydotplus.graph_from_dot_data(dot_data)
Image(graph3.create_png())
