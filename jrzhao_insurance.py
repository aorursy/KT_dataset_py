# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from scipy import stats
from scipy.special import boxcox1p
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.linear_model import LassoCV,RidgeCV
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
S = StandardScaler()
data = pd.read_csv("../input/insurance.csv")
data.head()

#data['age'] = np.log(1/data['age'])-1
a = np.log1p(data.age**2.7)
fig =plt.subplot(211)
sns.distplot(a.values, bins=10, kde=True,fit=norm)
plt.xlabel('charge', fontsize=12)
cs =plt.subplot(212)
res = stats.probplot(a, plot=plt)
plt.tight_layout()
plt.show()
data['bmi'] = boxcox1p(data['bmi'],0.8)
data['bmi'] = S.fit_transform(data['bmi'].values.reshape(-1, 1))
data.age = np.log1p(data.age**2.7)
data.charges = np.log1p(data.charges)
data['charges'] = S.fit_transform(data['charges'].values.reshape(-1, 1))
a = pd.DataFrame()
for i in ('sex','region','smoker'):
    a[i] = LabelEncoder().fit_transform(data[i].values)
a1 = OneHotEncoder(sparse = False).fit_transform(a[['sex','region','smoker']])
da = pd.DataFrame(columns = ('sex1','sex2','region1','region2','region3','region4','smoker1','smoker2'),data = a1)
data.drop(['sex','region','smoker'],axis = 1, inplace=True)
data = pd.concat([data,da],axis = 1)


X = data.drop(['charges'],axis = 1)
y = data['charges']
x_train, x_test ,y_train,y_test = train_test_split(X,y,random_state = 33 , test_size = 0.11)

alphas = [0.01]
lasso = LassoCV(alphas=alphas)
lasso.fit(x_train,y_train)
y_test_pred_lasso = lasso.predict(x_test)
y_train_pred_lasso = lasso.predict(x_train)
print('The r2 score of LassoCV on test is',r2_score(y_test, y_test_pred_lasso))
print('The r2 score of LassoCV on train is', r2_score(y_train, y_train_pred_lasso))
print('alpha is:',lasso.alpha_)

alphas = [2.2]
ridge = RidgeCV(alphas = alphas)
ridge.fit(x_train,y_train)
y_test_pred_ridge = ridge.predict(x_test)
y_train_pred_ridge = ridge.predict(x_train)

print('The r2 score of RidgeCV on test is ',r2_score(y_test, y_test_pred_ridge))
print('The r2 score of RidgeCV on train is', r2_score(y_train, y_train_pred_ridge))
print('alpha is:',ridge.alpha_)