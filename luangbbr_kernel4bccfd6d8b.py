import sklearn
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
import sklearn.naive_bayes as skNB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from IPython.display import Image
from IPython.core.display import HTML 
#leitura dataset de treino
train_raw = pd.read_csv("../input/atividade-3-pmr3508/train.csv", sep = ",")
train = train_raw.copy()
target = train_raw["median_house_value"]
display(train_raw.shape,
train_raw.head())
#leitura do dataset de teste
test_raw = pd.read_csv("../input/atividade-3-pmr3508/test.csv", sep = ",",header=0)
test = test_raw.copy()
display(test_raw.shape,test_raw.head())

def New_features(data):
    data.loc[:,'rooms_per_household'] = data.loc[:,'total_rooms']/data.loc[:,'households']
    data.loc[:,'rooms_per_person'] = data.loc[:,'total_rooms']/data.loc[:,'population']
    data.loc[:,'bedrooms_per_household'] = data.loc[:,'total_bedrooms']/data.loc[:,'households']
    data.loc[:,'bedrooms_per_person'] = data.loc[:,'total_bedrooms']/data.loc[:,'households']
    data.loc[:,'persons_per_household'] = data.loc[:,'population']/data.loc[:,'households']
    data.loc[:, 'median_income_per_person'] = data.loc[:,'median_income']/data.loc[:,'persons_per_household']
New_features(train_raw)
New_features(test)
New_features(train)
train.head()
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 12, 12  
pylab.rcParams['font.family'] = 'sans-serif'
pylab.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
pylab.rcParams['font.serif'] = ['Bitstream Vera Sans']
pylab.rcParams["font.size"] = "12"
plt.scatter(train_raw["longitude"],train_raw["latitude"],cmap='magma',c=(-1*np.log10(train_raw["median_house_value"])),s=2)
plt.axes().set_aspect('equal', 'datalim')

plt.show()
display(Image(url= "http://www.geocurrents.info/wp-content/uploads/2016/01/California-Median-Home-Price-Map.png"))
#tratamento da tabela de coordenadas das cidades americanas
arq = open("../input/us-citiestxt/US_cities.txt","r")

new_table=[]
linhas=arq.readlines()
for i in range(87,len(linhas)):
    palavras=linhas[i].split(",")
    if palavras[1] ==  "5":
        new_table.append([palavras[3][1:-1],palavras[5][:-1],palavras[4]])

arq=open("US_cities.csv","w")
for i in new_table:
    arq.write(i[0]+","+i[1]+","+i[2]+"\n")
#leitura da tabela de coordenadas
coords={}
arq=open("US_cities.csv")
for i in arq.readlines():
    lista=i.strip().split(",")
    coords[lista[0]]=(float(lista[1]),float(lista[2]))
coords
#treino
dSF_LA=[]
a=np.array(coords["San Francisco"])
b=np.array(coords["Los Angeles"])
for i,c in train.iterrows():
    c=np.array((c['longitude'],c['latitude']))
    dSF_LA.append(min(np.linalg.norm(a-c),np.linalg.norm(b-c)))
train = train.join(pd.Series(dSF_LA,name="dSF_LA"))
#teste
dSF_LA=[]
a=np.array(coords["San Francisco"])
b=np.array(coords["Los Angeles"])
for i,c in train.iterrows():
    c=np.array((c['longitude'],c['latitude']))
    dSF_LA.append(min(np.linalg.norm(a-c),np.linalg.norm(b-c)))
test = test.join(pd.Series(dSF_LA,name="dSF_LA"))

#lista criada por mim, das cidades praianas da califórnia
Coastal_Contys=["Del Norte", "Humboldt","Mendocino", "Sonoma", "Marin", "San Francisco", "San Mateo", "Santa Cruz", "Monterey", "San Luis Obispo", "Santa Barbara", "Ventura", "Los Angeles", "Orange", "San Diego"]

#treino
dSea=[]
for i,c in train.iterrows():
    b=np.array((c['longitude'],c['latitude']))
    distancias=[]
    for j in Coastal_Contys:
        a=np.array(coords[j])
        distancias.append(np.linalg.norm(a-b))
    dSea.append(min(distancias))
train = train.join(pd.Series(dSea,name="dSea"))

#teste
dSea=[]
for i,c in train.iterrows():
    b=np.array((c['longitude'],c['latitude']))
    distancias=[]
    for j in Coastal_Contys:
        a=np.array(coords[j])
        distancias.append(np.linalg.norm(a-b))
    dSea.append(min(distancias))
test = test.join(pd.Series(dSea,name="dSea"))

pylab.rcParams['figure.figsize'] = 6, 6
for i in train.columns:
    plt.scatter(train[i],target,s=1)
    plt.xlabel(i)
    plt.show()

    
train.head()
train.corr()["median_house_value"].abs().sort_values(ascending=False)
n=15
lista=list(train_raw.corr()["median_house_value"].abs().nlargest(n).keys())
trainF = train.reindex(lista,axis=1)
testF = test.reindex(lista,axis=1)
Id=test.loc[:,"Id"]
dfs=[trainF,testF]
vals=["median_house_value","Id"]
for i in dfs:
    for j in vals:
        try:
            i.drop(j,axis=1,inplace=True)
        except KeyError:
            pass
    
display("DATASET DE TESTE:",testF.head(),"DATASET DE TREINO:",trainF.head())
msle = make_scorer(mean_squared_log_error)
def score(nome,regressor):
    pontos = cross_val_score(regressor,trainF,target,cv=10,scoring=msle)
    print(nome+ ': '+ str(round(np.sqrt(pontos).mean(),4))+ "\n")
Lreg = Lasso(positive = True)
score("Regressor Lasso",Lreg)

RFreg = RandomForestRegressor(max_depth=20, random_state=0,n_estimators=400)
score("Regressor Random Forest",RFreg)
N = list(range(1,51))
pontuacao=[]
for i in N:
    neigh = KNeighborsRegressor(n_neighbors=i)
    notas = cross_val_score(neigh,trainF,target,cv=10,scoring=msle)
    pontuacao.append(np.sqrt(notas).mean())

display(plt.scatter(N,pontuacao,s=5))
print("Melhor K: "+ str(np.array(pontuacao).argmin()+1))
KNNreg = KNeighborsRegressor(n_neighbors=np.array(pontuacao).argmin()+1)
score("Regressor KNN",KNNreg)
Adareg = AdaBoostRegressor(RandomForestRegressor(max_depth=15,n_estimators=10),n_estimators=60)
#score("AdaBoost",Adareg)
Adareg.fit(trainF,target)
R = Adareg.predict(testF)
R
#predict
arq=open("result1.csv","w")
arq.write("Id,median_house_value\n")
for i,j in zip(list(R),list(Id)):
    arq.write(str(j)+","+str(i)+"\n")
arq.close()