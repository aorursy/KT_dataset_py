# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#データ格納

Data = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

Data.shape
Data.head()
#欠損値確認⇒name usd pledged

print(Data.isnull().any())
#必要のないものを除く

#stateは使ってもよい…？

data=Data.drop("name",1)#個人単位で異なる

data=data.drop("ID",1)#個人単位で異なる

data=data.drop("usd pledged",1)#結果として得られるため

data=data.drop("state",1)#結果として得られるため

data.head()
#時系列行（deadline）の変換

data['deadline2']=pd.to_datetime(data['deadline'])

data['deadline_y']=(data['deadline2'].dt.year)

data['deadline_m']=(data['deadline2'].dt.month)

data=data.drop("deadline",1)

data=data.drop("deadline2",1)

data.head()
#時系列行（launched）の変換

data['launched2']=pd.to_datetime(data['launched'])

data['launched_y']=(data['launched2'].dt.year)

data['launched_m']=(data['launched2'].dt.month)

data=data.drop("launched",1)

data=data.drop("launched2",1)

data.head()
#種類書き出してみる

u_category=data['category'].unique()

u_mcategory=data['main_category'].unique()

u_carrency=data['currency'].unique()

u_country=data['country'].unique()

print(u_category)

print(u_mcategory)

print(u_carrency)

print(u_country)
#ワンホットエンコード化

data2=pd.get_dummies(data,columns=['category'])

data2=pd.get_dummies(data2,columns=['main_category'])

data2=pd.get_dummies(data2,columns=['currency'])

data2=pd.get_dummies(data2,columns=['country'])

data2.head(10)
#usd_pledged_real / usd_goal_realの追加

data3=data2.assign(P = data2["usd_pledged_real"]/data2["usd_goal_real"])

data3=data3.drop("usd_pledged_real",1)

data3=data3.drop("usd_goal_real",1)

data3.head()
#標準偏差算出

data3.describe()
#相関係数算出

C=data3.corr()

C.head()
#説明変数

X=data3.drop("P",1)

X.head()

X.dropna()
#目的変数

Y=data3.P

Y.head(10)
#目的変数散布図行列

a=np.zeros((378661,1))

for i in range(378661):

    a[i,0]=i+1



import matplotlib.pyplot as plt

plt.scatter(a,Y)

plt.show()
#標準化

XX=((X-X.mean())/X.std())

XX.head()
#白色化定義

def white(X,epsilon=1e-5):

    n,p=X.shape

    u,v = np.linalg.eig(np.dot(X.T,X)/n)  #u固有値 vベクトル

    Z=np.dot(X,np.dot(v,np.diag(1/(np.sqrt(u)+epsilon))))

    return (Z)
#白色化

XXX=white(X,epsilon=1e-5)
#交差検証と重回帰分析

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

kf=KFold(n_splits=10,shuffle=True)



scores1 = cross_validate(lr,XX,Y,cv=kf,scoring={"neg_mean_squared_error","r2"})
#精度

print('Cross-Validation scores（平均二乗誤差・決定係数）:{}'.format(scores1))
#Ridge回帰

from sklearn.linear_model import Ridge

rg=Ridge()

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

kf=KFold(n_splits=5,shuffle=True)



scores1 = cross_validate(rg,XX,Y,cv=kf,scoring={"neg_mean_squared_error","r2"})
#精度

print('Cross-Validation scores（平均二乗誤差・決定係数）:{}'.format(scores1))
#Lasso回帰

from sklearn.linear_model import Lasso

ls=Lasso()

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

kf=KFold(n_splits=5,shuffle=True)



scores1 = cross_validate(ls,XX,Y,cv=kf,scoring={"neg_mean_squared_error","r2"})
#精度

print('Cross-Validation scores（平均二乗誤差・決定係数）:{}'.format(scores1))
#ElasticNet

from sklearn.linear_model import ElasticNet

en=ElasticNet()

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

kf=KFold(n_splits=5,shuffle=True)



scores1 = cross_validate(en,XX,Y,cv=kf,scoring={"neg_mean_squared_error","r2"})
#精度

print('Cross-Validation scores（平均二乗誤差・決定係数）:{}'.format(scores1))
#ElasticNetにて特徴量選択

from sklearn.linear_model import ElasticNet

estimator=ElasticNet()



from sklearn.feature_selection import RFECV

rfecv=RFECV(estimator,cv=3,scoring='neg_mean_absolute_error')
rfecv.fit(XX,Y)
#特徴量のランキング表示

print('Feature ranking: \n{}'.format(rfecv.ranking_))
# 特徴数とスコアの変化をプロット

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
rfecv.support_
remove_idx = ~rfecv.support_

remove_idx

remove_feature = XX.columns[remove_idx]

remove_feature

XX2 = XX.drop(remove_feature, axis=1)

XX2.head()
#ElasticNet

from sklearn.linear_model import ElasticNet

en=ElasticNet()

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

kf=KFold(n_splits=5,shuffle=True)



scores1 = cross_validate(en,XX2,Y,cv=kf,scoring={"neg_mean_squared_error","r2"})
#精度

print('Cross-Validation scores（平均二乗誤差・決定係数）:{}'.format(scores1))
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor()

kf=KFold(n_splits=5,shuffle=True)

scores1 = cross_validate(rfr,XX,Y,cv=kf,scoring={"neg_mean_squared_error","r2"})

print('Cross-Validation scores（平均二乗誤差・決定係数）:{}'.format(scores1))
#特徴量の重要度評価を行う

rfr=rfr.fit(X,Y)

feature=rfr.feature_importances_

print('特徴量:{}'.format(feature))
