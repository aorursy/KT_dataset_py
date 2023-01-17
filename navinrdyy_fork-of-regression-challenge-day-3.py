import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
camera = pd.read_csv('../input/1000-cameras-dataset/camera_dataset.csv')

camera.head(3)
camera = camera.set_index('Model')
pd.isnull(camera).any()
pd.isnull(camera).sum()
camera = camera.dropna()
pd.isnull(camera).any()
temp = camera.corr()
plt.subplots(figsize=(10,5))

sns.heatmap(temp, cmap='RdYlGn', annot=True)

plt.show()
camera['Avg resolution'] = (camera['Max resolution']+camera['Low resolution'])/2





camera['Avg zoom'] = ((camera['Zoom wide (W)']<=30.0)).astype(int)

camera['Avg zoom1'] = ((camera['Zoom tele (T)']<=120.0)).astype(int)



camera['Avg focus'] = (camera['Normal focus range']+camera['Macro focus range'])/2



camera.head(3)
camera['Avg resolution 1'] = (camera['Avg resolution']<=2000).astype(int)

camera['Avg resolution 2'] = ((camera['Avg resolution']>2000)&(camera['Avg resolution']<=2800)).astype(int)
camera['Epix 1'] = (camera['Effective pixels']<=4.0).astype(int)
camera['stor 1'] = (camera['Storage included']<=8.0).astype(int)

camera['stor 2'] = ((camera['Storage included']>8.0) & camera['Storage included']<15.0).astype(int)

camera['stor 3'] = ((camera['Storage included']>=15.0) & camera['Storage included']<=64.0).astype(int)
camera['Weight 1'] = (camera['Weight (inc. batteries)']<=180.0).astype(int)

camera['Weight 2'] = ((camera['Weight (inc. batteries)']>180.0) & (camera['Weight (inc. batteries)']<=320.0)).astype(int)

camera['Weight 3'] = ((camera['Weight (inc. batteries)']>320.0) & (camera['Weight (inc. batteries)']<=1100.0)).astype(int)
camera['Dim 1'] = (camera['Dimensions']<=40.0).astype(int)

camera['Dim 2'] = ((camera['Dimensions']>40.0)&(camera['Dimensions']<=80.0)).astype(int)

camera['Dim 3'] = ((camera['Dimensions']>80.0)&(camera['Dimensions']<=130.0)).astype(int)
labels = ['0-750','750-1500','1500-1900']

camera['Weight_bins'] = pd.cut(camera['Weight (inc. batteries)'],3,right=True,labels=labels)
camera['Weight_bins 1'] = (camera['Weight_bins']== '0-750').astype(int)

camera['Weight_bins 2'] = (camera['Weight_bins']== '1000-1500').astype(int)
labels = ['0-30','30-60','60-90']

camera['Macro_bins'] = pd.cut(camera['Macro focus range'],3,right=True,labels=labels)
camera['Macro_bins 1'] = (camera['Macro_bins']== '0-30').astype(int)

camera['Macro_bins 2'] = (camera['Macro_bins']== '30-60').astype(int)
labels = ['0-25','25-35','35-53']

camera['Zoom_w_bins'] = pd.cut(camera['Zoom wide (W)'],3,right=True,labels=labels)
camera['Zoom_w_bins 1'] = (camera['Zoom_w_bins']== '0-25').astype(int)

camera['Zoom_w_bins 2'] = (camera['Zoom_w_bins']== '25-35').astype(int)
camera.head(3)
X = camera.drop(['Release date','Price','Effective pixels','Avg focus','Normal focus range','Macro focus range','Epix 1','Storage included','Dimensions','Low resolution','Weight (inc. batteries)','Weight_bins','Macro_bins','Zoom_w_bins'],axis=1)

y = camera['Price']
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state=1,test_size=0.19)

lin = LinearRegression()

lin.fit(Xtrain,ytrain)
y_pred = lin.predict(Xtest)
np.sqrt(metrics.mean_squared_error(ytest,y_pred))
df = pd.DataFrame({})

df['Price'] = ytest

df['Predicted'] = y_pred

df['ERROR'] = df['Price'] - df['Predicted']

df.head(15)
df['ERROR'].describe()
lin.intercept_
lin.coef_
plt.subplots(figsize=(10,5))

plt.scatter(df['Price'],df['ERROR'],color='black')

plt.xlabel('Price',fontsize=18,fontweight='bold',color='navy')

plt.ylabel('Error',fontsize=18,fontweight='bold',color='navy')

plt.show()