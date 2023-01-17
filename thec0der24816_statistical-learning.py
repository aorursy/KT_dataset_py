# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

credit_d=pd.read_csv('../input/Credit.csv')
credit_d.describe()
credit_d.info()
#Limiting columns and mapping features
X_c=credit_d.iloc[:,[1,2,3,8]].values
Y_c=credit_d.iloc[:,11].values
X_c
Y_c
credit_d["Student"].value_counts()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X_c[:,3] = labelencoder.fit_transform(X_c[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X_c = onehotencoder.fit_transform(X_c).toarray()
X_c=X_c[:,1:]
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_c,Y_c, test_size = 0.2, random_state= 42)
X_train
Y_train
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
from sklearn.metrics import mean_squared_error
L_MSE=mean_squared_error(Y_test,Y_pred)
L_RMSE=np.sqrt(L_MSE)
L_MSE
L_RMSE
reg.intercept_
reg.coef_
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
#Feature scaling 
from sklearn.preprocessing import StandardScaler
ssc=StandardScaler()
X_train_s=ssc.fit_transform(X_train)
X_test_s=ssc.fit_transform(X_test)

# MinMax
from sklearn.preprocessing import MinMaxScaler
mmxs=MinMaxScaler()
X_train_mmx=mmxs.fit_transform(X_train)
X_test_mmx=mmxs.fit_transform(X_test)
Rreg=Ridge(alpha=2).fit(X_train_s,Y_train)
#X_test_s
Rreg.intercept_
Rreg.coef_
Rreg.score(X_train_s,Y_train)

Rreg.score(X_test_s,Y_test)
# Ridge Regression
for alpha1 in [0.1,0.5,1,2,5,10,20,50,100,200,500,1000]:
    Rrega=Ridge(alpha=alpha1).fit(X_train_s,Y_train)
    R2_train=Rrega.score(X_train_s,Y_train)
    R2_test=Rrega.score(X_test_s,Y_test)
    num_coeff=np.sum(abs(Rrega.coef_)>1)
    print('\n Alpha = {:.2f}\n Intercept={}\n Coefficient={}\n Non zero coeff:{},\n r-squared training:{:,.2}\n r-squared test:{:,.2}'.format(alpha1,Rrega.intercept_,Rrega.coef_,num_coeff,R2_train,R2_test))
# Ridge coefficient as a function of alpha
alpha_n=200
alphas=np.logspace(-5,5,alpha_n)
coefs=[]

for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_train_s,Y_train)
    coefs.append(ridge.coef_)
    
ax=plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale('Log')
ax.set_xlim(ax.get_xlim()[::1])
ax.set_ylim(ax.get_ylim()[::1])

plt.xlabel('Alpha')
plt.ylabel('Weights')
plt.title('Ridge coefficient by regularization')
plt.axis('tight')
plt.show()