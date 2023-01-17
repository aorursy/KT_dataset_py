import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline



from matplotlib import rcParams

rcParams['figure.figsize']=12,8

data = pd.read_csv("../input/diamonds.csv")
data.head()
data.drop(data.columns[0],axis=1,inplace=True)
data.isnull().sum()
data.cut.value_counts()
data.color.value_counts()
data.clarity.value_counts()
plt.plot(data.carat,data.price,'.',color='red')

plt.show()
plt.plot(data.depth,data.price,'.',color='red')

plt.show()
plt.plot(data.table,data.price,'.',color='red')

plt.show()
plt.plot(data.depth,data.table,'.',color='red')

plt.show()
plt.plot(data.x,data.price,'.',color='red')

plt.show()
plt.plot(data.y,data.price,'.',color='red')

plt.show()
plt.plot(data.z,data.price,'.',color='red')

plt.show()
import seaborn as sns
sns.heatmap(data.corr(),annot=True)

plt.show()
sns.pairplot(data)

plt.show()
y = data['price']
del data['price']
sns.violinplot(data.color,y)

plt.show()
sns.violinplot(data.clarity,y)

plt.show()
sns.violinplot(data.cut,y)

plt.show()
data['color'].replace(['D','E','F','G','H','I','J'],[6,5,4,3,2,1,0],inplace=True)
data['cut'].replace(['Fair','Good','Very Good','Premium','Ideal'],[0,1,2,3,4],inplace=True)
data.clarity.value_counts()
data['clarity'].replace(['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'],[0,1,2,3,4,5,6,7],inplace=True)
x=data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler
scl=StandardScaler()
x_train_scaled = scl.fit_transform(x_train)
x_test_scaled = scl.transform(x_test)
from sklearn.decomposition import PCA
pca=PCA()
pca.fit(x_train_scaled)
pca.explained_variance_ratio_
var=np.cumsum(pca.explained_variance_ratio_)
plt.plot(var,color='red')

plt.show()
pca=PCA(n_components=5)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train_pca,y_train)
lr.coef_
y_train_pred = lr.predict(x_train_pca)
y_test_pred = lr.predict(x_test_pca)
from sklearn.metrics import r2_score,mean_squared_error
r2_score(y_train,y_train_pred)
r2_score(y_test,y_test_pred)
plt.plot(y_train_pred, y_train-y_train_pred,'.',color='blue')

plt.show()
plt.plot(y_test_pred, y_test-y_test_pred,'.',color='blue')

plt.show()
for i in range(5):

    plt.plot(x_train_pca[:,i],y_train,'.',color='red')

    plt.show()
from math import sqrt
x2 = np.sqrt(abs(x_train_pca[:,0]))
plt.plot(x2,y_train,'.')

plt.show()



plt.plot(x_train_pca[:,0],y_train,'.')

plt.show()
x_train_pca[:,0] = x2
lr.fit(x_train_pca,y_train)
lr.coef_
y_train_pred = lr.predict(x_train_pca)
r2_score(y_train,y_train_pred)
# square root transformation makes the predictions worse
# gbm
from sklearn.ensemble import GradientBoostingRegressor
gbm=GradientBoostingRegressor()
gbm.fit(x_train,y_train)
y_train_pred=gbm.predict(x_train)
r2_score(y_train,y_train_pred)
y_test_pred = gbm.predict(x_test)
r2_score(y_test,y_test_pred)
plt.plot(y_train_pred,y_train-y_train_pred,'.',color='red')

plt.show()
plt.plot(y_test_pred,y_test-y_test_pred,'.',color='red')

plt.show()
feat_imp=pd.Series(gbm.feature_importances_,x_train.columns).sort_values(ascending=False)
feat_imp.plot(kind='bar')

plt.show()
#  !All good!