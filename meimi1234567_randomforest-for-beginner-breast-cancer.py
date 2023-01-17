from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV,cross_val_score

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

data = load_breast_cancer()
data.data.shape

#569樣本，30個特徵
data.target
rfc = RandomForestClassifier(n_estimators=100,random_state=90) #實例化

score_pre = cross_val_score(rfc,data.data,data.target,cv = 10).mean()

score_pre
#畫n_estimators的學習曲線先從取每十个数作为一个阶段

scorel = []

for i in range(0,200,10): #會以1、11、21...一路到200

    rfc = RandomForestClassifier(n_estimators=i+1, n_jobs=-1,random_state=90)#n_estimators不為零

    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()

    scorel.append(score) 

print(max(scorel),(scorel.index(max(scorel))*10)+1) #返回scorel中最大值的索引＊10＋1(因為間距是10，n_estimators有+1)

plt.figure(figsize=[20,5]) 

plt.plot(range(1,201,10),scorel)

plt.show()



#返回出：得出來scorel的最大值是，n_estimators是
#縮短範圍（一樣的步驟）

scorel = []

for i in range(65,75): 

    rfc = RandomForestClassifier(n_estimators=i, n_jobs=-1,random_state=90)

    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()

    scorel.append(score) 

print(max(scorel)

      ,([*range(65,75)][scorel.index(max(scorel))]))  #返回scorel中最大值的索引 

plt.figure(figsize=[20,5]) 

plt.plot(range(65,75),scorel)

plt.show()
#網格搜索

#调整max_depth

param_grid = {'max_depth':np.arange(1, 20, 1)}#希望搜索的參數

rfc = RandomForestClassifier(n_estimators=73  #根據上面的調n_estimators的結果

                             ,random_state=90

                            )

GS = GridSearchCV(rfc,param_grid,cv=10).fit(data.data,data.target)      

GS.best_params_ #顯示最佳調整出來的參數
GS.best_score_#返回最大參數的準確率
#调整max_features

param_grid = {'max_features':np.arange(5,30,1)} 

#max_features的默認的最小值是特徵數的開根號，一直調到特徵數能到的最大值

rfc = RandomForestClassifier(n_estimators=73

                             ,random_state=90)

GS = GridSearchCV(rfc,param_grid,cv=10).fit(data.data,data.target) 

GS.best_params_
GS.best_score_

#還是降低惹
#调整min_samples_leaf(默認是1)

param_grid = {'min_samples_leaf':np.arange(1,1+10,1)} 

#max_features的默認的最小值是特徵數的開根號，一直調到特徵數能到的最大值

rfc = RandomForestClassifier(n_estimators=73

                             ,random_state=90)

GS = GridSearchCV(rfc,param_grid,cv=10).fit(data.data,data.target)

GS.best_params_ #如果=默認值有調根沒調一樣
GS.best_score_
#调整min_samples_split(默認值是2)

param_grid = {'min_samples_split':np.arange(2,2+20,1)} 



rfc = RandomForestClassifier(n_estimators=73

                             ,random_state=90)

GS = GridSearchCV(rfc,param_grid,cv=10).fit(data.data,data.target) 

GS.best_params_ #如果等於默認值有調根沒調一樣
GS.best_score_
param_grid = {'criterion':['gini', 'entropy']}

rfc = RandomForestClassifier(n_estimators=73

                             ,random_state=90

                            )

GS = GridSearchCV(rfc,param_grid,cv=10)

GS.fit(data.data,data.target)

GS.best_params_ 
GS.best_score_
rfc = RandomForestClassifier(n_estimators=73,random_state=90) 

score = cross_val_score(rfc,data.data,data.target,cv=10).mean() 

score

score - score_pre