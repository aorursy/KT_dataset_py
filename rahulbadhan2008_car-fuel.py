import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

from plotly.offline import download_plotlyjs,init_notebook_mode,iplot,plot

init_notebook_mode(connected=True)

cfl=pd.read_csv('../input/car-consume/measurements.csv')

cfl['distance']=cfl['distance'].str.replace(',','.').astype(float)

cfl['consume']=cfl['consume'].str.replace(',','.').astype(float)

cfl['temp_inside']=cfl['temp_inside'].str.replace(',','.').astype(float)

cfl
cfl.info()
cfl=cfl.drop(['specials','refill liters','refill gas'],axis=1)

cfl
cfl.info()
cfl['temp_inside']=cfl['temp_inside'].fillna(16.0)

cfl.describe()
sns.pairplot(cfl)

plt.show()
cfl['speed'].plot.hist()

plt.show()
cfl['distance'].plot.hist()

plt.show()
sns.countplot(cfl['AC'])

plt.show()
maxd=cfl.query('distance==216.100000')

maxd
sns.lmplot('distance','consume',data=maxd)

plt.show()
mind=cfl.query('distance==1.300000')

mind
sns.lmplot('distance','consume',data=mind)

plt.show()
maxs=cfl.query('speed==90.000000')

maxs
sns.lmplot('distance','consume',data=maxs)

plt.show()
mins=cfl.query('speed==14.000000')

mins
sns.lmplot('distance','consume',data=mins)

plt.show()
minr=cfl.query('rain==0.000000')

minr
sns.lmplot('distance','consume',data=minr)

plt.show()
maxr=cfl.query('rain==1.000000')

maxr
sns.lmplot('distance','consume',data=maxr)

plt.show()
minsn=cfl.query('sun==0.000000')

minsn
sns.lmplot('distance','consume',data=minsn)

plt.show()
minsn=cfl.query('sun==1.000000')

minsn
sns.lmplot('distance','consume',data=minsn)

plt.show()
minac=cfl.query('AC==0.000000')

minac
sns.lmplot('distance','consume',data=minac)

plt.show()
maxac=cfl.query('AC==1.000000')

maxac
sns.lmplot('distance','consume',data=maxac)

plt.show()
sns.countplot(cfl['gas_type'])

plt.show()
gas1=cfl[cfl['gas_type']=='E10']

gas1
gas1.describe()
gas1mnd=gas1.query('distance==1.700000')

gas1mnd
sns.lmplot('distance','consume',data=gas1mnd)

plt.show()
gas1mxd=gas1.query('distance==130.300000')

gas1mxd
sns.lmplot('distance','consume',data=gas1mxd)

plt.show()
gas1mxs=gas1.query('speed==88.000000')

gas1mxs
sns.lmplot('speed','consume',data=gas1mxs)

plt.show()
gas1mns=gas1.query('speed==14.000000')

gas1mns
sns.lmplot('speed','consume',data=gas1mns)

plt.show()
gas2=cfl[cfl['gas_type']=='SP98']

gas2
gas2.describe()
gas2mnd=gas2.query('distance==1.300000')

gas2mnd
sns.lmplot('distance','consume',data=gas2mnd)

plt.show()
gas2mxd=gas2.query('distance==216.100000')

gas2mxd
sns.lmplot('distance','consume',data=gas2mxd)

plt.show()
gas2mxs=gas2.query('speed==90.000000')

gas2mxs
sns.lmplot('speed','consume',data=gas2mxs)

plt.show()
gas2mns=gas2.query('speed==16.000000')

gas2mns
sns.lmplot('speed','consume',data=gas2mns)

plt.show()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

cfl['gas_type']=le.fit_transform(cfl['gas_type'])

cfl['distance']=cfl['distance'].astype(int)

cfl['consume']=cfl['consume'].astype(int)

cfl['temp_inside']=cfl['temp_inside'].astype(int)
x=cfl.drop(['consume','AC'],axis=1)



y=cfl['consume']



x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.40,random_state=43)
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

x_trainst=ss.fit_transform(x_train)

x_testst=ss.transform(x_test)
from sklearn.preprocessing import MinMaxScaler

mms=MinMaxScaler()

x_trainmn=mms.fit_transform(x_train)

x_testmn= mms.transform(x_test)
from sklearn.decomposition import PCA

pc=PCA(n_components=4)

x_trainpca=pc.fit_transform(x_train)

x_testpca=pc.transform(x_test)
from sklearn.tree import DecisionTreeRegressor

dt1=DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=3,random_state=72)

dt1.fit(x_train,y_train)

py1=dt1.predict(x_test)

from sklearn import metrics

import numpy as np

from sklearn.metrics import mean_squared_error

from math import sqrt

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py1)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py1))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py1))

plt.scatter(y_test,py1)

plt.xlabel('y_test(True Values)')

plt.ylabel('py')

plt.show()
sns.distplot((y_test-py1),bins=50)

plt.show()
ftr=['distance','speed','gas_type','rain','temp_inside','temp_outsid','sun']



from sklearn.tree import export_graphviz

export_graphviz(dt1,out_file="tree.dot",feature_names=ftr,rounded=True,filled=True)

from subprocess import call

call(['dot','-Tpng','tree.dot','-o','tree1.png','-Gdpi=600'])



from IPython.display import Image

Image(filename='tree1.png')
from sklearn.tree import DecisionTreeRegressor

dt2=DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=3,random_state=72)

dt2.fit(x_trainst,y_train)

py2=dt2.predict(x_testst)

from sklearn import metrics

import numpy as np

from sklearn.metrics import mean_squared_error

from math import sqrt

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py2)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py2))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py2))

plt.scatter(y_test,py2)

plt.xlabel('y_test(True Values)')

plt.ylabel('py')

plt.show()
sns.distplot((y_test-py2),bins=50)

plt.show()
ftr=['distance','speed','gas_type','rain','temp_inside','temp_outsid','sun']



from sklearn.tree import export_graphviz

export_graphviz(dt2,out_file="tree.dot",feature_names=ftr,rounded=True,filled=True)

from subprocess import call

call(['dot','-Tpng','tree.dot','-o','tree2.png','-Gdpi=600'])



from IPython.display import Image

Image(filename='tree2.png')
from sklearn.tree import DecisionTreeRegressor

dt3=DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=3,random_state=72)

dt3.fit(x_trainmn,y_train)

py3=dt3.predict(x_testmn)

from sklearn import metrics

import numpy as np

from sklearn.metrics import mean_squared_error

from math import sqrt

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py3)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py3))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py3))

plt.scatter(y_test,py3)

plt.xlabel('y_test(True Values)')

plt.ylabel('py')

plt.show()
sns.distplot((y_test-py3),bins=50)

plt.show()
ftr=['distance','speed','gas_type','rain','temp_inside','temp_outsid','sun']



from sklearn.tree import export_graphviz

export_graphviz(dt3,out_file="tree.dot",feature_names=ftr,rounded=True,filled=True)

from subprocess import call

call(['dot','-Tpng','tree.dot','-o','tree3.png','-Gdpi=600'])



from IPython.display import Image

Image(filename='tree3.png')
from sklearn.tree import DecisionTreeRegressor

dt4=DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=3,random_state=72)

dt4.fit(x_trainpca,y_train)

py4=dt4.predict(x_testpca)

from sklearn import metrics

import numpy as np

from sklearn.metrics import mean_squared_error

from math import sqrt

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py4)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py4))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py4))

plt.scatter(y_test,py4)

plt.xlabel('y_test(True Values)')

plt.ylabel('py')

plt.show()
sns.distplot((y_test-py4),bins=50)

plt.show()
ftrpc=['distance','speed','gas_type','rain']



from sklearn.tree import export_graphviz

export_graphviz(dt4,out_file="tree.dot",feature_names=ftrpc,rounded=True,filled=True)

from subprocess import call

call(['dot','-Tpng','tree.dot','-o','tree4.png','-Gdpi=600'])



from IPython.display import Image

Image(filename='tree4.png')
from sklearn.ensemble import RandomForestRegressor

md5=RandomForestRegressor(n_estimators='warn',criterion='mse', random_state=44, max_depth=3)

md5.fit(x_train,y_train)

py5=md5.predict(x_test)

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py5)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py5))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py5))
plt.scatter(y_test,py5)

plt.xlabel('y_test(True Values)')

plt.ylabel('py5')

plt.show()
sns.distplot((y_test-py5),bins=50)

plt.show()
tree = md5.estimators_[5]

from sklearn.tree import export_graphviz

import pydot

export_graphviz(tree, out_file = 'tree.dot', feature_names = ftr, rounded = True, precision = 1,filled=True)

(graph,) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('tree5.png')

import matplotlib.image as mpimg

plt.imshow(mpimg.imread('tree5.png'))

plt.show()
from sklearn.ensemble import RandomForestRegressor

md6=RandomForestRegressor(n_estimators='warn',criterion='mse', random_state=44, max_depth=3)

md6.fit(x_trainst,y_train)

py6=md6.predict(x_testst)



print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py6)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py6))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py6))
plt.scatter(y_test,py6)

plt.xlabel('y_test(True Values)')

plt.ylabel('py6')

plt.show()

sns.distplot((y_test-py6),bins=50)

plt.show()
tree = md6.estimators_[5]

from sklearn.tree import export_graphviz

import pydot

export_graphviz(tree, out_file = 'tree.dot', feature_names = ftr, rounded = True, precision = 1,filled=True)

(graph,) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('tree6.png')

import matplotlib.image as mpimg

plt.imshow(mpimg.imread('tree6.png'))

plt.show()
from sklearn.ensemble import RandomForestRegressor

md7=RandomForestRegressor(n_estimators='warn',criterion='mse', random_state=44, max_depth=3)

md7.fit(x_trainmn,y_train)

py7=md7.predict(x_testmn)



print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py7)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py7))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py7))
plt.scatter(y_test,py7)

plt.xlabel('y_test(True Values)')

plt.ylabel('py7')

plt.show()

sns.distplot((y_test-py7),bins=50)

plt.show()
tree = md7.estimators_[5]

from sklearn.tree import export_graphviz

import pydot

export_graphviz(tree, out_file = 'tree.dot', feature_names = ftr, rounded = True, precision = 1,filled=True)

(graph,) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('tree7.png')

import matplotlib.image as mpimg

plt.imshow(mpimg.imread('tree7.png'))

plt.show()
from sklearn.ensemble import RandomForestRegressor

md8=RandomForestRegressor(n_estimators='warn',criterion='mse', random_state=44, max_depth=3)

md8.fit(x_trainpca,y_train)

py8=md8.predict(x_testpca)



print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py8)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py8))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py8))
plt.scatter(y_test,py8)

plt.xlabel('y_test(True Values)')

plt.ylabel('py8')

plt.show()

sns.distplot((y_test-py8),bins=50)

plt.show()
tree = md8.estimators_[5]

from sklearn.tree import export_graphviz

import pydot

export_graphviz(tree, out_file = 'tree.dot', feature_names = ftrpc, rounded = True, precision = 1,filled=True)

(graph,) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('tree8.png')

import matplotlib.image as mpimg

plt.imshow(mpimg.imread('tree8.png'))

plt.show()
from sklearn.linear_model import LinearRegression

lnr=LinearRegression()

lnr.fit(x_train,y_train)

py9=lnr.predict(x_test)

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py9)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py9))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py9))
plt.scatter(y_test,py9)

plt.xlabel('y_test(True Values)')

plt.ylabel('py')

plt.show()
print('Variation Score:',metrics.explained_variance_score(y_test,py9))
sns.distplot((y_test-py9),bins=50)

plt.show()
from sklearn.linear_model import LinearRegression

lnr=LinearRegression()

lnr.fit(x_trainst,y_train)

py10=lnr.predict(x_testst)

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py10)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py10))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py10))
plt.scatter(y_test,py10)

plt.xlabel('y_test(True Values)')

plt.ylabel('py')

plt.show()
sns.distplot((y_test-py10),bins=50)

plt.show()
from sklearn.linear_model import LinearRegression

lnr=LinearRegression()

lnr.fit(x_trainmn,y_train)

py11=lnr.predict(x_testmn)

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py11)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py11))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py11))
plt.scatter(y_test,py10)

plt.xlabel('y_test(True Values)')

plt.ylabel('py')

plt.show()
sns.distplot((y_test-py10),bins=50)

plt.show()
from sklearn.linear_model import LinearRegression

lnr=LinearRegression()

lnr.fit(x_trainpca,y_train)

py12=lnr.predict(x_testpca)

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, py12)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, py12))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, py12))
plt.scatter(y_test,py12)

plt.xlabel('y_test(True Values)')

plt.ylabel('py')

plt.show()
sns.distplot((y_test-py12),bins=50)

plt.show()