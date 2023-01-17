from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt  # Matlab-style plotting
from sklearn.metrics import mean_squared_error

import os
print(os.listdir("../input"))

treino = pd.read_csv("../input/train.csv") #base de treino completa, com os preços e os ids
x_treino = treino.drop(columns=["Id","median_house_value"])
y_treino = treino["median_house_value"]


teste = pd.read_csv("../input/test.csv") #base de teste completa, com os ids
x_teste = teste.drop(columns = "Id") #atributos da base teste, sem ids

teste_positions = teste[["latitude","longitude"]]
treino_positions = treino[["latitude","longitude"]]

x_treino.describe()
treino_si = treino.drop(columns="Id")
tab = treino_si.corr(method="spearman")
df = pd.DataFrame(tab)

def pinta(x):
    if abs(x) == 1:
        color = 'black'
    elif abs(x) >= abs(df.quantile(q=0.75).mean()):
        color = 'indianred' 
    elif abs(x) >= abs(df.mean().mean()):
        color = 'aqua'
    elif abs(x) <= abs(df.mean().mean())/6:
        color = 'white'
    elif abs(x) <= abs(df.mean().mean())/3:
        color = 'paleturquoise' 
    elif abs(x) <= abs(df.mean().mean()):
        color = 'dodgerblue'
    else:
        color = 'white'
            
    return 'background-color: %s' % color
    
print("Salmon : correlação maior que a média do 3º quartil das correlações ")
print("Azul mais vivo, aqua : correlação maior que a média das correlações ")
print("Azul mais escuro, dodgerblue : correlação pequena ")
print("Azul mais clara, paleturquoise : correlação menor que um terço da média")
print("Branco : correlação menor que um sexto da média ")
df.style.applymap(pinta)

fig, ax = plt.subplots()
ax.scatter(x = treino['median_income'], y = treino['median_house_value'] , s = 1)
plt.ylabel('median_house_value', fontsize=13)
plt.xlabel('median_income', fontsize=13)
plt.title("Relacionamento entre renda e valor do imóvel")
plt.show()

x_treino["median_age"].hist()
x_teste["median_age"].hist()
plt.ylabel('frequência', fontsize=13)
plt.xlabel('median_age', fontsize=13)
plt.title("Histograma de idades médias das regiões; laranja teste e azul treino")


plt.subplots()
plt.scatter(x = treino['median_age'], y = treino['median_house_value'], s=0.5)
plt.ylabel('median_house_value', fontsize=13)
plt.xlabel('median_age', fontsize=13)
plt.title("Relacionamento entre idade média e valor do imóvel")
plt.show()

plt.subplots()
plt.scatter(treino["longitude"],treino["latitude"], c= treino["median_age"] , cmap = "jet" , s = 5)
plt.title("Idade média por posição geográfica; quanto mais quente a cor, maior a idade")
plt.ylabel('latitude', fontsize=13)
plt.xlabel('longitude', fontsize=13)
plt.xlim(-130,-110)
plt.show()
from IPython.display import Image
from IPython.core.display import HTML 

plt.subplots
plt.xlim(-130,-110)
plt.plot(teste["longitude"],teste["latitude"],"g.")
plt.show()

plt.subplots
plt.xlim(-130,-110)
plt.scatter(treino["longitude"],treino["latitude"], c= treino["median_house_value"] , cmap = "jet" , s = 10)
plt.show()

Image(url= "http://trackurls.info/wp-content/uploads/2018/02/a-stylized-map-of-the-state-showing-different-big-cities-pacific-ocean-and-nearby-states-california-with-counties.jpg",width=400)

from geopy import distance
sacramento = (38.575764,-121.478851) #posição geográfica da capital californiana
losangeles = (34.0522342, -118.2436849) #posição geográfica de Los Angeles

def tratar(data):
    
    nlinhas = (data.shape[0])
    
    data["distance_sacramento"] = [0] * nlinhas
    data["distance_la"] = [0] * nlinhas
    
    #data["latitude_region"] = [0] * nlinhas
    #dh = data["longitude"]
    #data["litoral"] = [0] * nlinhas
    
    for i in range(nlinhas):
        
        data.loc[(i,"distance_sacramento")] = distance.distance((data.loc[i, 'latitude'] , data.loc[i, 'longitude']),sacramento).km
        data.loc[(i,"distance_la")] = distance.distance((data.loc[i, 'latitude'] , data.loc[i, 'longitude']), losangeles).km

#        if data["latitude_region_number"][i] >= data["latitude_region_number"].quantile(q=0.75): 
#            data.loc[(i,"latitude_region")] = 8 #região extremo sul da califórnia, ~LA, San Diego
#            if -119 < dh[i] <= -117:
#                data.loc[(i,"litoral")] = 1
            
            
#        elif data["latitude_region_number"][i] >= data["latitude_region_number"].quantile(q=0.5): 
#            data.loc[(i,"latitude_region")] = 6 #região centro sul, ~Fresno
#            if -119.5 < dh[i] <= -118.5:
#                data.loc[(i,"litoral")] = 1
            
            
#        elif data["latitude_region_number"][i] >= data["latitude_region_number"].quantile(q=0.25):
#            data.loc[(i,"latitude_region")] = 4 #região centro norte, ~Sacramento
#            if -123 < dh[i] <= -119.5:
#                data.loc[(i,"litoral")] = 1            
            
        
#        elif data["latitude_region_number"][i] >= 1:
#            data.loc[(i,"latitude_region")] = 1 #região extremo norte
#            if -125 <= dh[i] <= -122.7:
#                data.loc[(i,"litoral")] = 1            

               
    #criando colunas úteis
    data["razao_quarto_comodo"] = data["total_bedrooms"] / data["total_rooms"]
    data["razao_comodo_imovel"] = data["total_rooms"] / data["households"]
    data["razao_pop_house"] = data["population"] / data["households"]
    data["renda_per_capita"] = data["median_income"] / data["population"]
    
    data.drop(columns=["population","total_rooms","total_bedrooms","median_age","latitude","longitude"], inplace = True) #não preciso mais dessas colunas
    


#x_treino["latitude_region_number"].describe()
#count    14448.000000
#mean         1.094495
#std          0.065705
#min          1.000000
#25%          1.042396
#50%          1.052227
#75%          1.158525
#max          1.288786
tratar(x_treino)
tabe = x_treino.corr(method="spearman")
tabe.style.applymap(pinta)
tratar(x_teste)
x_teste.shape[1] == x_treino.shape[1] #verificando dimensões
#### Método Lasso  ---- etapa 1 boosting

las = linear_model.Lasso(alpha=0.6)
las = las.fit(x_treino, y_treino)
pred = las.predict(x_treino)

print("Cross-value-score:  ",las.score(x_treino,y_treino, sample_weight=None))

print("Erro quadrático médio:  " ,mean_squared_error(y_treino, pred))

#### Método Ridge

rid = linear_model.RidgeCV(alphas=[0.1 ,0.8,0.3], cv=7)
rid = rid.fit(x_treino,y_treino)
pred = rid.predict(x_treino)

print("Cross-value-score:  ",rid.score(x_treino,y_treino, sample_weight=None))

print("Erro quadrático médio:  " ,mean_squared_error(y_treino, pred))

from catboost import CatBoostRegressor

gato = CatBoostRegressor(learning_rate=1, depth=4,num_trees= 20, loss_function='RMSE')
fit_gato = gato.fit(x_treino, y_treino)
predigato = gato.predict(x_treino)


print("Erro quadrático médio:  " ,mean_squared_error(y_treino, predigato))

y_gato = abs (gato.predict(x_teste))
smit = pd.DataFrame({"Id": teste["Id"] , "median_house_value": y_gato[:]})
smit.to_csv("_gatoboost_", index = False)
#### Método K Nearest Neighbors com 80 vizinhos

knn = KNeighborsRegressor(n_neighbors = 80, weights = "distance")
knn.fit( x_treino ,  y_treino)

print("Cross-value-score:  ",knn.score(x_treino,y_treino, sample_weight=None))
print("Erro quadrático médio:  " ,mean_squared_error(y_treino, knn.predict(x_treino)))

knn30 = KNeighborsRegressor(n_neighbors = 30, weights = "distance")
### Random Forest:
'''
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=30, criterion='gini', max_depth= 2 , 
                                    max_features='auto', min_impurity_decrease=0.3,
                                    bootstrap=True, oob_score=False, warm_start=False, n_jobs = -1)

forest.fit(x_treino,y_treino)


print("Cross-value-score:  ",forest.score(x_treino,y_treino, sample_weight=None))

print("Erro quadrático médio:  " , mean_squared_error(y_treino, forest.predict(x_treino)))
'''


### Regression Tree::
from sklearn.tree import DecisionTreeRegressor

arvore = DecisionTreeRegressor(criterion="mse", max_depth=20, min_samples_split=6, 
                               min_samples_leaf=8, min_impurity_decrease=0.2, 
                               presort=True)

arvore.fit(x_treino, y_treino)
pred = arvore.predict(x_treino)

print("Cross-value-score:  ",arvore.score(x_treino,y_treino, sample_weight=None))

print("Erro quadrático médio:  " ,mean_squared_error(y_treino, pred))

y_arvore = arvore.predict(x_teste)
smit = pd.DataFrame({"Id": teste["Id"] , "median_house_value": y_arvore[:]})
smit.to_csv("_arvore_", index = False)

### LDA ::

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
pred = lda.fit(x_treino, y_treino).predict(x_treino)

print("Cross-value-score:  ",lda.score(x_treino,y_treino, sample_weight=None))

print("Erro quadrático médio:  " ,mean_squared_error(y_treino, pred))

### QDA :: 

#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#qda = QuadraticDiscriminantAnalysis().fit(x_treino, y_treino)

#print("Cross-value-score:  ",qda.score(x_treino,y_treino, sample_weight=None))

#print("Erro quadrático médio:  " ,mean_squared_error(y_treino, qda.predict(xx)))


#smit = pd.DataFrame({"Id": teste["Id"] , "median_house_value": predtest[:]})
#smit.to_csv("sub_QDA", index = False)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor

bagging = BaggingRegressor(knn30, max_samples=0.3, max_features=0.3)
bagpred = bagging.fit(x_treino,y_treino).predict(x_teste)


adatree = AdaBoostRegressor(arvore, n_estimators=100)
adapred = adatree.fit(x_treino,y_treino).predict(x_teste)


smita = pd.DataFrame({"Id": teste["Id"] , "median_house_value": bagpred[:]})
smita.to_csv("_bolsa_", index = False)

smitb = pd.DataFrame({"Id": teste["Id"] , "median_house_value": adapred[:]})
smitb.to_csv("_ada_", index = False)

#tentativa 1

las = las.fit(x_treino, y_treino)
pred_in = las.predict(x_treino)
erro1 = y_treino - pred_in

prederro1 = adatree.fit(x_treino, erro1).predict (x_treino)
erro2 = erro1 - prederro1

knn30.fit( x_treino ,  erro2)
prederro2 = knn30.predict(x_treino)
erro3 = erro2 - prederro2

for i in range (5,45,8):
    print(y_treino[i] ,'=', pred_in[i] , prederro1[i] , prederro2[i])

print()
for i in range(5,45,8):
    print(erro1[i],prederro1[i], 'und', erro2[i], prederro2[i])
#para base teste

y_teste_primario = las.predict(x_teste)
prederroa = adatree.predict(x_teste)
prederrob = knn30.predict (x_teste)

y_teste = y_teste_primario + prederroa + prederrob

smition = pd.DataFrame({"Id": teste["Id"] , "median_house_value": y_teste[:]})
smition.to_csv("_boost_4", index = False)
#tentativa 2 

erroi = y_treino - predigato

prederroi = adatree.fit(x_treino, erroi).predict (x_treino)
erroii = erroi - prederroi

prederroii = knn30.fit( x_treino ,  erroii).predict(x_treino)
erroiii = erroii - prederroii

for i in range (5,45,8):
    print(y_treino[i] ,'=', predigato[i] , prederroi[i] , prederroii[i] , erroiii[i])

print()
for i in range(5,45,8):
    print(erroi[i],prederroi[i], 'und', erroii[i], prederroii[i])
#para base teste da tentativa 2

y_teste_primario = gato.predict(x_teste)
prederroa = arvore.predict(x_teste)
prederrob = knn30.predict (x_teste)

y_teste = y_teste_primario + prederroa + prederrob

smition = pd.DataFrame({"Id": teste["Id"] , "median_house_value": y_teste[:]})
smition.to_csv("_boost_5", index = False)
#tentativa 3 =============

pred_in = lda.fit(x_treino, y_treino).predict(x_treino)
erroi = y_treino - pred_in

prederroi = adatree.fit(x_treino, erroi).predict (x_treino)
erroii = erroi - prederroi

prederroii = knn30.fit( x_treino ,  erroii).predict(x_treino)
erroiii = erroii - prederroii


for i in range (5,45,8):
    print(y_treino[i] ,'=', pred_in[i] , prederroi[i] , prederroii[i] , erroiii[i])

print()
for i in range(5,45,8):
    print(erroi[i],prederroi[i], 'und', erroii[i], prederroii[i])
#para base teste da tentativa 3

y_teste_primario = lda.predict(x_teste)
prederroa = adatree.predict(x_teste)
prederrob = knn30.predict (x_teste)

y_teste = y_teste_primario + prederroa + prederrob

smition = pd.DataFrame({"Id": teste["Id"] , "median_house_value": y_teste[:]})
smition.to_csv("_boost_6", index = False)