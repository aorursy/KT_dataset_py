import pandas as pd
import numpy as np
data=pd.read_csv("../input/challenge/Train_data.csv").dropna()
data = data.sample(frac=1).reset_index(drop=True)
d1=data[data['accident']==0].head(1500)
d2=data[data['accident']==1].head(607)
train=pd.concat([d1,d2])
data.isna().sum()
drop=['Unnamed: 0','gender','code']#+['planName','registrationMode','clientType']
train.drop(columns=drop,inplace=True)
(data['accident']==0).sum()
(data['accident']==1).sum()
from sklearn.model_selection import train_test_split
y=train.accident
X=train.drop(columns=['accident'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,shuffle=True)
from sklearn.preprocessing import LabelEncoder
train=X_train.astype(str)
enc=LabelEncoder()
train=train.apply(enc.fit_transform)
train=pd.DataFrame(train)
x_train=train
# from sklearn.preprocessing import MinMaxScaler
# scale=MinMaxScaler()
# train[train.columns]=pd.DataFrame(scale.fit_transform(train))
from sklearn.linear_model import BayesianRidge
dt=BayesianRidge(n_iter=50, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=True, fit_intercept=True, normalize=True, copy_X=True, verbose=False)
dt.fit(x_train,y_train)
test=pd.read_csv("../input/challenge/Test_Data.csv")
test.drop(columns=drop,inplace=True)
test=test.astype(str)
test=test.apply(enc.fit_transform)
res=pd.DataFrame(columns=['accident']) 
res['accident']=dt.predict(test).round(1)
res.index+=1
res.to_csv('put17.txt',index=False)
res.describe()

