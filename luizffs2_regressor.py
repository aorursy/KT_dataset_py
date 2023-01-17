import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import cross_val_score as cvs
arquivo1 = '../input/test.csv'
tester = pd.read_csv(arquivo1, engine = 'python')
tester.shape
arquivo2 = '../input/train.csv'
trainer = pd.read_csv(arquivo2, engine = 'python')
trainer.shape
trainer.head()
target = trainer['median_house_value']
class Dados:
    pass
dt = Dados()        
dt.Id = list(trainer['Id'])
dt.lo = list(trainer['longitude'])
dt.la = list(trainer['latitude'])
dt.ma = list(trainer['median_age'])
dt.tr = list(trainer['total_rooms'])
dt.tb = list(trainer['total_bedrooms'])
dt.pp =list(trainer['population'])
dt.hh = list(trainer['households'])
dt.mi = list(trainer['median_income'])
dt.mhv = list(trainer['median_house_value'])
plt.plot(trainer['median_income'],trainer['median_house_value'],'bo')
dt.pph = []
for i in range(len(dt.pp)):
    dt.pph.append(dt.hh[i]/dt.pp[i])
dt.ppr = []
for i in range(len(dt.pp)):
    dt.ppr.append(dt.tr[i]/dt.pp[i])
dt.ppb = []
for i in range(len(dt.pp)):
    dt.ppb.append(dt.tb[i]/dt.pp[i])
dt.bph = []
for i in range(len(dt.hh)):
    dt.bph.append(dt.tb[i]/dt.hh[i])
plt.plot(dt.pph, dt.mhv, 'bo')
plt.plot(dt.bph, dt.mhv, 'bo')
plt.plot(dt.ppr, dt.mhv, 'bo')
plt.plot(dt.ppb, dt.mhv, 'bo')
plt.plot(dt.ma, dt.mhv, 'bo')
plt.plot(dt.hh, dt.mhv, 'bo')

plt.plot(dt.lo, dt.la, 'bo')
x = []
y = []
for i in range(len(dt.mhv)):
    if dt.mhv[i] > 300000:
        y.append(dt.la[i])
        x.append(dt.lo[i])
plt.plot(x, y, 'bo')
train = trainer[['latitude', 'longitude', 'median_income', 'households']]
def dist_from(lat1,lon1,lat2,lon2):
    #Função que calcula a distancia entre 2 pontos
    return ((lat1-lat2)**2+(lon1-lon2)**2)**0.5
dt2 = Dados()
dt2.lo = list(tester['longitude'])
dt2.la = list(tester['latitude'])
sc = [38.5666, -121.469]
sd = [32.8153, -117.135]
sf = [37.7272, -123.032]
sj = [37.2969, -121.819]
ti = [34.0194, -118.411]
la = [32.533, -117.05]
#Base de treino
dt.dsc = []
dt.dsd = []
dt.dsf = []
dt.dsj = []
dt.dti = []
dt.dla = []
for i in range(len(dt.lo)):
    dt.dsc.append(dist_from(dt.la[i], dt.lo[i], sc[0], sc[1]))
    dt.dsd.append(dist_from(dt.la[i], dt.lo[i], sd[0], sd[1]))
    dt.dsf.append(dist_from(dt.la[i], dt.lo[i], sf[0], sf[1]))
    dt.dsj.append(dist_from(dt.la[i], dt.lo[i], sj[0], sj[1]))
    dt.dti.append(dist_from(dt.la[i], dt.lo[i], ti[0], ti[1]))
    dt.dla.append(dist_from(dt.la[i], dt.lo[i], la[0], la[1]))
#Base de testes
dt2.dsc = []
dt2.dsd = []
dt2.dsf = []
dt2.dsj = []
dt2.dti = []
dt2.dla = []
for i in range(len(dt2.lo)):
    dt2.dsc.append(dist_from(dt.la[i], dt.lo[i], sc[0], sc[1]))
    dt2.dsd.append(dist_from(dt.la[i], dt.lo[i], sd[0], sd[1]))
    dt2.dsf.append(dist_from(dt.la[i], dt.lo[i], sf[0], sf[1]))
    dt2.dsj.append(dist_from(dt.la[i], dt.lo[i], sj[0], sj[1]))
    dt2.dti.append(dist_from(dt.la[i], dt.lo[i], ti[0], ti[1]))
    dt2.dla.append(dist_from(dt.la[i], dt.lo[i], la[0], la[1]))
train.insert(4, 'dsc', dt.dsc)
train.insert(5, 'dsf', dt.dsf)
train.insert(6, 'dsj', dt.dsj)
train.insert(7, 'dti', dt.dti)
train.insert(8, 'dla', dt.dla)
test = tester[['latitude', 'longitude', 'median_income', 'households']]
test.insert(4, 'dsc', dt2.dsc)
test.insert(5, 'dsf', dt2.dsf)
test.insert(6, 'dsj', dt2.dsj)
test.insert(7, 'dti', dt2.dti)
test.insert(8, 'dla', dt2.dla)
def melhor_knr(base,p,u):
    '''Recebe a base de testes, o primeiro e o último número de nearest neighbors a ser 
       verificado e retorna a melhor acurácia obtida bem como o número de nearest neighbors
       utilizado
    '''
    score = 0
    for i in range(p,u):
        knr = KNR(n_neighbors=i)
        new_scores = cvs(knr,train,target,cv=10)
        if new_scores.mean() > score:
            score = new_scores.mean()
            nearn = i
            
    return score, nearn
best_knr = KNR(n_neighbors=melhor_knr(train,5,50)[1])
best_knr.fit(train,trainer['median_house_value'])
knr_pred = best_knr.predict(test)
Submit1 = pd.DataFrame()
Submit1.insert(0, 'Id', tester['Id'])
Submit1.insert(1,'median_house_value', knr_pred)
file = open('knr_pred.csv','w')
file.write(pd.DataFrame.to_csv(Submit1, index=False))
file.close()
def melhor_arvore(base,p,u):
    '''Recebe a base de testes, o primeiro e o último número de nearest neighbors a ser 
       verificado e retorna a melhor acurácia obtida bem como o número de nearest neighbors
       utilizado
    '''
    score = 0
    for i in range(p,u):
        dtr = DTR(max_depth=i)
        new_scores = cvs(dtr,train,target,cv=10)
        if new_scores.mean() > score:
            score = new_scores.mean()
            best_tree = i
            
    return score, best_tree
melhor_arvore(train,5,20)
best_tree = DTR(max_depth=10)
best_tree.fit(train,target)
tree_pred = best_tree.predict(test)
tree_pred
Submit2 = pd.DataFrame()
Submit2.insert(0, 'Id', tester['Id'])
Submit2.insert(1,'median_house_value', tree_pred)
file = open('tree_pred.csv','w')
file.write(pd.DataFrame.to_csv(Submit2, index=False))
file.close()
lcv = LassoCV().fit(train, target)
lcv.score(train, target)
lasso = Lasso(max_iter = 100000, selection = 'random')
lasso.fit(train, target)
pred_lasso = lasso.predict(test)
pred_lasso
Submit3 = pd.DataFrame()
Submit3.insert(0, 'Id', tester['Id'])
Submit3.insert(1,'median_house_value', pred_lasso)
file = open('lasso_pred.csv','w')
file.write(pd.DataFrame.to_csv(Submit3, index=False))
file.close()