#imports padrões
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore') 

#carregando os datasets
features=pd.read_csv("/kaggle/input/features.csv")
samples=pd.read_csv("/kaggle/input/sampleSubmission.csv")
stores=pd.read_csv("/kaggle/input/stores.csv")
test=pd.read_csv("/kaggle/input/test.csv")
train=pd.read_csv("/kaggle/input/train.csv") 
print("features\n")
features.info()
print("\nlojas\n")
stores.info()
print("\ntrain\n")
train.info()
print("\ntest\n")
test.info()
stores.head(10)
features.head(10)
train.head(10)
test.head(10)
train.tail()
#para verificar se o teste é a continuação do train
#verificando como é a distribuição dos departamentos pelas lojas:
departamentos=[]
for i in range(1,46): #numero de lojas
    print("quantidade de departamentos na loja",i,":",train[train["Store"]==i]["Dept"].nunique())
    departamentos.append(train[train["Store"]==i]["Dept"].nunique())
print ("\nquantidade máxima e mínima de departamentos (respectivamente):",max(departamentos),min(departamentos))
#verificando se todas as lojas tem todas as features    
#temos 8190 dados. Divido por 45 = 182. Para que os dados sejam completos, os len tem que ser todos iguais a 182
len_features=len(features[features["Store"]==1])
print ("quantidade de dados por lojas:",len_features)
lista_diferentes=[]
for i in range(2,46):
    if len(features[features["Store"]==i]) != len_features:
        lista_diferentes.append(i)
print ("quantidade de lojas com features diferentes:" ,len(lista_diferentes)) #se for 0, todas são iguais
#verificando a quantidade de dados nulos

#Store           8190 non-null int64
#Date            8190 non-null object
#Temperature     8190 non-null float64
#Fuel_Price      8190 non-null float64
#MarkDown1       4032 non-null float64
#MarkDown2       2921 non-null float64
#MarkDown3       3613 non-null float64
#MarkDown4       3464 non-null float64
#MarkDown5       4050 non-null float64
#CPI             7605 non-null float64
#Unemployment    7605 non-null float64
#IsHoliday       8190 non-null bool

print("CPI null % = {:.2f}%".format((len(features[features["CPI"].isnull()])/8190)*100))
print("Desemprego null % = {:.2f}%".format((len(features[features["Unemployment"].isnull()])/8190)*100))
print("MarkDown1 null % = {:.2f}%".format((len(features[features["MarkDown1"].isnull()])/8190)*100))
print("MarkDown2 null % = {:.2f}%".format((len(features[features["MarkDown2"].isnull()])/8190)*100))
print("MarkDown3 null % = {:.2f}%".format((len(features[features["MarkDown3"].isnull()])/8190)*100))
print("MarkDown4 null % = {:.2f}%".format((len(features[features["MarkDown4"].isnull()])/8190)*100))
#verificando se os dados que faltam em CPI e Desemprego são dos mesmos dias
(features[features["CPI"].isnull()==True].groupby("Date").sum()) == (features[features["Unemployment"].isnull()==True].groupby("Date").sum())
#criando um df com todas as informações (features, vendas, tamanho e tipo das lojas), para procurar correlação

#adicionando o tamanho e tipo das lojas
storesComFeatures=pd.merge(stores,features, on="Store",how="inner")

#adicionando as vendas
treino_dados_totais=pd.merge(storesComFeatures,train,how="inner",on=["Store","Date","IsHoliday"]).reset_index(drop=True)
treino_dados_totais.sort_values(by=["Store","Dept","Date"])
teste_dados_totais=pd.merge(storesComFeatures,test,how="inner",on=["Store","Date","IsHoliday"]).reset_index(drop=True)
teste_dados_totais.sort_values(by=["Store","Dept","Date"])

#corrigindo o formato da data e transformando Holiday para booleano
treino_dados_totais["Date"]=     pd.to_datetime(treino_dados_totais["Date"])
treino_dados_totais["IsHoliday"]=pd.get_dummies(treino_dados_totais["IsHoliday"],drop_first=True)
teste_dados_totais["Date"]=      pd.to_datetime(teste_dados_totais["Date"])
teste_dados_totais["IsHoliday"]= pd.get_dummies(teste_dados_totais["IsHoliday"],drop_first=True)

#além disto, vou criar uma coluna com o ano e uma com a semana, para explorar a sazonalidade das vendas
treino_dados_totais["ano"]=   treino_dados_totais["Date"].dt.year
treino_dados_totais["semana"]=treino_dados_totais["Date"].dt.week
teste_dados_totais["ano"]=    teste_dados_totais["Date"].dt.year
teste_dados_totais["semana"]= teste_dados_totais["Date"].dt.week

#transformando o tipo da loja em numérico:
treinoDummies=pd.get_dummies (treino_dados_totais["Type"],prefix="Store_Type")
testeDummies= pd.get_dummies (teste_dados_totais["Type"],prefix="Store_Type")

treino_dados_totais=pd.concat([treino_dados_totais, treinoDummies], axis=1)
teste_dados_totais= pd.concat([teste_dados_totais, testeDummies], axis=1)

treino_dados_totais.drop("Type",axis=1,inplace=True)
teste_dados_totais.drop ("Type",axis=1,inplace=True)

#Como os Markdowns são promoções, faz sentido trocar os valores nulos por 0, pois indica a ausência de promoção
treino_dados_totais["MarkDown4"].fillna(0,inplace=True)
treino_dados_totais["MarkDown5"].fillna(0,inplace=True)
teste_dados_totais["MarkDown4"].fillna(0,inplace=True)
teste_dados_totais["MarkDown5"].fillna(0,inplace=True)

del storesComFeatures,features,stores,test,train   #liberando memória

treino_dados_totais.head()
teste_dados_totais.head()
#construindo datasets por loja para plotar:
dados_totais=[]
data=treino_dados_totais[treino_dados_totais["Store"]==i].reset_index().groupby(["Date"]).sum().reset_index()['Date']
for i in range(1,46):
    dados_totais.append(
        treino_dados_totais[treino_dados_totais["Store"]==i].reset_index().groupby(["Date"]).sum().reset_index()["Weekly_Sales"])
dados_totais=pd.DataFrame(dados_totais).T
dados_totais.columns=(np.arange(1,46))
dados_totais["Date"]=data
#plots das vendas por loja ao longo do tempo
plt.figure(figsize=(30,20))
#dividindo em 9 plots com 5 loja cada
for i in range(1,10):   
    for j in range(1,6):   
        plt.subplot(3,3,i)
        plt.plot(dados_totais["Date"],dados_totais[i*j],lw=1,alpha=0.5)
del dados_totais,data
#pairplot, para uma primeira visão geral
sns.pairplot(treino_dados_totais)
#heatmap
correl=treino_dados_totais.corr().abs()
plt.figure(figsize=(17,15))
sns.heatmap(correl,square=True,annot=True,cbar_kws={"shrink":0.7})
#investigando os dados nulos:
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,7))
ax[0].scatter(treino_dados_totais["CPI"],treino_dados_totais["Weekly_Sales"],color="red",marker="1",lw=1)
ax[1].scatter(treino_dados_totais["Unemployment"],treino_dados_totais["Weekly_Sales"],color="green",marker="+",lw=1)
ax[0].set_title("Vendas x CPI")
ax[1].set_title("Vendas x Unemployment")
ax[0].set_xlabel("CPI")
ax[0].set_ylabel("Vendas")
ax[1].set_xlabel("Unemployment")
ax[1].set_ylabel("Vendas")
teste_dados_totais.drop(["Fuel_Price","MarkDown1","MarkDown2","MarkDown3","Temperature","CPI"],axis=1,inplace=True)
treino_dados_totais.drop(["Fuel_Price","MarkDown1","MarkDown2","MarkDown3","Temperature","CPI"],axis=1,inplace=True)
#plot ao longo do tempo
temp=teste_dados_totais[teste_dados_totais["Store"]==1].groupby(["Date"]).mean()
temp.reset_index(inplace=True)
plt.figure(figsize=(15,5))
plt.plot(temp["Date"],temp["Unemployment"])
del temp   #liberando memória
#pegando a média e o desvio pradrão dos dados de desemprego no perídodo
print("Média =",teste_dados_totais[teste_dados_totais["Unemployment"].isna()==False].groupby(["Date"]).mean()["Unemployment"].mean())
print("Desvio Padrão =",teste_dados_totais[teste_dados_totais["Unemployment"].isna()==False].groupby(["Date"]).mean()["Unemployment"].std())
teste_dados_totais["Unemployment"].fillna(method="ffill",inplace=True) 
#um novo plot de correlações, após as alterações e deleções de colunas
correl=treino_dados_totais.corr().abs()
plt.figure(figsize=(17,15))
sns.heatmap(correl,square=True,annot=True,cbar_kws={"shrink":0.7})
#outro pairplot, para ficar mais claro
sns.pairplot(treino_dados_totais)
#removendo a data
teste_final=teste_dados_totais.drop("Date",axis=1)
teste_final.sort_values(by=["Store","Dept"],inplace=True)  #para ficar do formato correto da submission
treino_final=treino_dados_totais.drop("Date",axis=1)

#criando a função WMAE, como indicado no enunciado
def WMAE(df, real, pred):
    peso = df["IsHoliday"].apply(lambda x: 5 if x else 1)
    return np.round(np.sum(peso*abs(real-pred))/(np.sum(peso)), 2)
#os modelos de previsão utilizados serão:
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest= train_test_split(
    treino_final.drop("Weekly_Sales",axis=1) , treino_final["Weekly_Sales"] , test_size=0.3, random_state=42)

#loop para os parâmetros da Random Forest:
print ("Método = RF")
min_erro_rf=1665
for n in [200,500,1000]:
    for depth in [10,50,100]:
        for leaf in [1,2,5]:
            for split in [2,5,10]:
                rf=RandomForestRegressor(n_estimators=n,
                                         max_depth=depth,
                                         min_samples_split=split,
                                         min_samples_leaf=leaf,
                                         random_state=42)
                rf.fit(xtrain,ytrain)
                pred=rf.predict(xtest)
                erro=WMAE(xtest,ytest,pred)
                print("n=",n,",depth=",depth,",leaf=",leaf,",split=",split)
                print("Erro =",erro)
                if erro<min_erro_rf:
                    min_erro_rf=erro
                    melhor_param_rf=[n,depth,leaf,split]
print ("Menor erro=",min_erro_rf)
print ("Melhores parametros=",melhor_param_rf)
#loop para os parâmetros do KNN:
print ("Método = KNN")
min_erro_knn=9981
for n in [1,3,5,7,10,15,50,100,200,500]:
    for weights in ["uniform", "distance"]:
        knn=KNeighborsRegressor(n_neighbors=n,
                               weights=weights,
                               n_jobs=-1)
        knn.fit(xtrain,ytrain)
        pred=knn.predict(xtest)
        erro=WMAE(xtest,ytest,pred)
        print("n=",n,",weights=",weights)
        print("Erro =",erro)
        if erro<min_erro_knn:
            min_erro_knn=erro
            melhor_param_knn=[n,weights]
print ("Menor erro",min_erro_knn)
print ("Melhores parametros=",melhor_param_knn)
#loop para os parâmetros da MLP
print("Método = MLP")
min_erro_mlp=12719
for size in [(50,),(100,),(200,),(500,)]:
    nn=MLPRegressor(hidden_layer_sizes=size,
                    random_state=42)
    nn.fit(xtrain,ytrain)
    pred=nn.predict(xtest)
    erro=WMAE(xtest,ytest,pred)
    print("size=",size)
    print("Erro =",erro,"\n\n")
    if erro<min_erro_mlp:
        min_erro_mlp=erro
        melhor_param_mlp=size
print ("Menor erro",min_erro_mlp)
print ("Melhores parametros=",melhor_param_mlp)
#este acabou não sendo rodado
#loop para os parâmetros do SVM
print("Método = SVM")
min_erro_svm=13799
for c in [0.1,1,10,100]:
    for gamma in [0.001,0.01,0.1,1]:
        svm=SVR(C=c,
               gamma=gamma)
        svm.fit(xtrain,ytrain)
        pred=svm.predict(xtest)
        erro=WMAE(xtest,ytest,pred)
        print("size=",size)
        print("Erro =",erro)
        if erro<min_erro_svm:
            min_erro_svm=erro
            melhor_param_svm=size
print ("Menor erro=",min_erro_svm)
print ("Melhores parametros=",melhor_param_svm)
#fazendo a submissão
#melhor método entre os 4: Random Forest
melhor_param_rf = [500, 100, 1, 2]
rf=RandomForestRegressor(n_estimators=melhor_param_rf[0],
                        max_depth=melhor_param_rf[1],
                        min_samples_leaf=melhor_param_rf[2],
                        min_samples_split=melhor_param_rf[3],
                        random_state=42)
rf.fit(xtrain,ytrain)
previsto=rf.predict(teste_final)
samples["Weekly_Sales"]=previsto
samples.to_csv('submission.csv',index=False)