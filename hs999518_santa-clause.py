import pandas as pd

import numpy as np

fd=pd.read_csv("../input/santa-2019-revenge-of-the-accountants/family_data.csv",encoding='ISO-8859-1')

fd

ss=pd.read_csv("../input/santa-2019-revenge-of-the-accountants/sample_submission.csv",encoding='ISO-8859-1')

ss

data=pd.merge(fd,ss,on='family_id',how='inner')

data

X=data.iloc[:,5:11]

y=data.loc[:,['assigned_day']]

X

y

data



data=pd.merge(fd,ss,on='family_id',how='inner')

data

a=data.n_people.min()



a

ag=data.groupby('n_people')['family_id'].count().sort_values(ascending=True)

ag

import matplotlib.pyplot as plt

plt.bar(ag.index,ag.values)

plt.show()



b=data.groupby('assigned_day')['family_id'].count().sort_values(ascending=True)

b

plt.bar(b.index,b.values,width=1)

plt.show()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=0)

print(X_train)

print(y_train)

print(X_test)

print(y_test)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)

pred=lr.predict(X_test)

pred

from sklearn import metrics

from math import sqrt

from sklearn.metrics import mean_squared_error

print("RMSE is:")

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, pred))

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred)))
from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor(random_state = 100,max_depth=2)

decision_tree.fit(X_train, y_train)

predictValues =decision_tree.predict(X_test)



predictValues





from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")



print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, predictValues))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, predictValues))

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictValues)))





data_feature_names = ['choice_4','choice_5','choice_6','choice_7','choice_8','choice_9']



from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(decision_tree, out_file=None, feature_names=data_feature_names, filled = True,rounded=True))





display(SVG(graph.pipe(format='svg')))
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=10, random_state=50, max_depth=5)



model.fit(X_train, y_train)

pred1=model.predict(X_test)

pred1



from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")



print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, pred1))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pred1))

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred1)))



estimators=model.estimators_[5]

labels=['choice_4','choice_5','choice_6','choice_7','choice_8','choice_9']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None

   , feature_names=labels

   , filled = True))

display(SVG(graph.pipe(format='svg')))

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_trainNorm2=ss.fit_transform(X_train)

X_testNorm2=ss.transform(X_test)

X_trainNorm2

X_testNorm2



from sklearn.decomposition import PCA

pa=PCA(n_components=2)

X_trainNorm=pa.fit_transform(X_trainNorm2)

X_trainNorm

X_testNorm=pa.transform(X_testNorm2)

X_testNorm



print(pa.explained_variance_ratio_)





from sklearn.preprocessing import MinMaxScaler

mms=MinMaxScaler()

X_trainNorm1=mms.fit_transform(X_train)

X_testNorm1=mms.transform(X_test)

X_trainNorm1

X_testNorm1

#using PCA

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_trainNorm,y_train)

pred2=lr.predict(X_testNorm)

pred2



from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:")

print('Root Mean Square Error:',np.sqrt(metrics.mean_squared_error(y_test,pred2)))
from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor(random_state = 100,max_depth=2)

decision_tree.fit(X_trainNorm, y_train)

predictValues1 =decision_tree.predict(X_testNorm)



predictValues1





from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")



print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictValues1)))
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=10, random_state=50, max_depth=5)



model.fit(X_trainNorm, y_train)

pred3=model.predict(X_testNorm)

pred3



from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")





print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred3)))
#USing MinMax Scaler

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_trainNorm1,y_train)

pred4=lr.predict(X_testNorm1)

pred4



from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:")

print('Root Mean Square Error:',np.sqrt(metrics.mean_squared_error(y_test,pred4)))
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=10, random_state=50, max_depth=5)



model.fit(X_trainNorm1, y_train)

pred5=model.predict(X_testNorm1)

pred5



from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")





print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred5)))
from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor(random_state = 100,max_depth=2)

decision_tree.fit(X_trainNorm1, y_train)

predictValues2 =decision_tree.predict(X_testNorm1)



predictValues2





from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")



print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictValues2)))

#using Standard Scaler

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_trainNorm2,y_train)

pred6=lr.predict(X_testNorm2)

pred6



from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:")

print('Root Mean Square Error:',np.sqrt(metrics.mean_squared_error(y_test,pred6)))
from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor(random_state = 100,max_depth=2)

decision_tree.fit(X_trainNorm2, y_train)

predictValues3 =decision_tree.predict(X_testNorm2)



predictValues3





from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")



print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predictValues3)))

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=10, random_state=50, max_depth=5)



model.fit(X_trainNorm2, y_train)

pred7=model.predict(X_testNorm2)

pred7



from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print("RMSE is:-")





print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred7)))


plt.plot(y_test,pred4,color = 'red')

plt.show()