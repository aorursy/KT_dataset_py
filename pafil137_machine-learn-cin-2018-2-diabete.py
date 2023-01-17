########################BIBLIOTECAS################################
#MLP
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

#Math AUX
import math
import numpy as np

#DATA USER
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

#Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score

#PLOT GRAPH
import matplotlib.pyplot as plt
########################ENTRADAS###########################
#CONFIG: testes
solverC     = ["adam", "sgd",      "adam", "sgd",      "sgd",  "sgd"]
activationC = ["relu", "logistic", "tanh", "identity", "relu", "tanh"]
config = 0

#VALOR MINIMO DA CLASSE
minClass = 4
percent  = 0.05

#CONFIG
qtd_batch = "auto"#(200, n_samples)
solver =  solverC[config]
activation = activationC[config]
hidden_layer=(100,100,100,100,)
Epochs = 1000 #max
learning_rate = 0.005
momentumAll = [0]
########################DADOS######################################
#TREINO E TESTE
data = pd.read_csv("../input/treino.csv")
dataT = pd.read_csv("../input/teste.csv")

#Avaliar dados
data.hist(bins=50, figsize=(20,15))
plt.show()

#Avalia a representatividade dos dados para as CLASSES
plt.figure(figsize=[10,4])
sb.heatmap(data.corr())
plt.show()
#SAIDA DA CLASSIFICACAO
out = "class_count"

#REMOVER DATA POR CLASSES 4+
data.drop (data[data[out]   < minClass].index ,inplace=True)
dataT.drop(dataT[dataT[out] < minClass].index ,inplace=True)

###################CONJUNTOS ENTRADA SAIDA
#TREINO
Yall =  data[out].copy()
Xall =  data.drop(out,axis=1)

#TESTE
YTall =  dataT[out].copy()
XTall =  dataT.drop(out,axis=1)

#CONJ. TREINO/TESTE
concatY = pd.concat([Yall, YTall], ignore_index=True, sort=False)
concatX = pd.concat([Xall, XTall], ignore_index=True, sort=False)
########################### NORMALIZAÇÃO: PREPARAR DADOS
#DADOS LABEL(CATEGORICAS - CAT) / DADOS NUMERICOS 
cat_atrib     = ["race","gender","age", "payer_code","medical_specialty", "max_glu_serum","A1Cresult","metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose","miglitol","troglitazone","tolazamide","examide","citoglipton","insulin","glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone","change","diabetesMed","readmitted"]
numeric_atrib = ["admission_type_id", "discharge_disposition_id", "admission_source_id", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient","number_emergency", "number_inpatient","diag_1","diag_2","diag_3","number_diagnoses"]

print("categoricas: ", len(cat_atrib))
print("numericas: ", len(numeric_atrib))

def normalizeData(data):
    ##normalizar dados NUMERICOS
    numeric_data = data[numeric_atrib]
    numeric_data.replace("NaN",0)
    imputer = Imputer(strategy='median')
    imputer.fit(numeric_data)
    numeric_data_complete = imputer.transform(numeric_data)
    numeric_data_normalized = StandardScaler().fit_transform(numeric_data_complete)
    
    ###normalizar dados CATEGORICOS
    categoric_data = data[cat_atrib]
    null_columns=categoric_data.columns[categoric_data.isnull().any()]
    categoric_data[categoric_data.isnull().any(axis=1)][null_columns].head()
    categoric_data_encoded = pd.get_dummies(categoric_data)
    
    data_norm = np.concatenate((categoric_data_encoded.values, numeric_data_normalized),1 )
    
    return data_norm

#X- normalizado
Xall = normalizeData(concatX)

#Y- vetorizado
Ypart = np.array(Yall) 
Ypart = Ypart.flatten()

Yall = np.array(concatY) 
Yall = Yall.flatten()

#DIVIDIR TESTE
#testPercent= 0.2
testPercent= 0.01

#ENTRADAS
#X = Xall[:int(len(Xall)*(1 - testPercent))]
X = Xall
Xt   = Xall[int(len(Xall)*(1 - testPercent)):]

#SAIDAS
#y  = Yall[:int(len(Yall)*(1 - testPercent))]
y = Yall
yt = Yall[int(len(Yall)*(1 - testPercent)):]

print("TREINO:",X.shape)
print("TESTE:", Xt.shape)
##PROCESSAMENTO##############################
#Gerar treino entrada e saida DNN
def criarMLPC():
    print("####CRIANDO MLPC####")
    
    for momentum in momentumAll:
        mlp = MLPClassifier(solver=solver, 
                           hidden_layer_sizes = hidden_layer,
                           activation=activation,
                           learning_rate_init=learning_rate, 
                           random_state=1,
                           batch_size=qtd_batch, 
                           momentum=momentum,
                           max_iter=Epochs)
        
        #TREINO
        accurace  = []
        precision = []
        bloss = 99999
        baccurace  = 0
        i = 0
        n_classes = np.unique(y)
        
        while (bloss > percent) and (i < Epochs):
            i+=1
            mlp.partial_fit(X, y, n_classes)
            pred = mlp.predict(X)
            
            baccurace  = accuracy_score(y, pred)
            bloss = 1 - precision_score(y, pred, average='weighted')
            
            accurace.append(baccurace)
            precision.append(bloss)
            
            print(i," ACC:",baccurace," Loss", bloss)
        
        #TESTE
        pred = mlp.predict(Xt)
        
        baccurace  = accuracy_score(yt, pred)
        bloss = 1 - precision_score(yt, pred, average='weighted')
        print("*************\n ACC:",baccurace," Loss", bloss,"\n*************")
            
        #ANALIZAR A SAIDA
        debugMLPC(pred, accurace, precision)
##SAIDA############################## 
def debugMLPC(pred=[], acc=[], loss=[]):
             
    #MSE - EVOLUTION
    plt.plot(acc)
    plt.plot(loss)
    plt.legend("AL")
    plt.title("ACCURACE VS LOSS")
    plt.show()

    #PREDICTED VS RIGHT
    plt.plot(yt)
    plt.plot(pred, 'ro')
    plt.legend("RP")
    plt.title("Right vs Predicted values")
    plt.show()
    
    #Graph1: Resultado Bruto
    plt.plot(sorted(pred), 'ro')
    plt.plot(sorted(yt), "b+")
    plt.legend("RP")
    plt.title("Real Values vs Predicted Points")
    plt.show()    
    
if __name__ == '__main__':     
    criarMLPC()