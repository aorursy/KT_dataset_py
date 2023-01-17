# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import statistics
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/iris/Iris.csv")
df=df.iloc[:,1:]
y=df['Species']
x=df.iloc[:,:4]
#Pré-processamento
encoder = LabelEncoder()
encoder.fit(y)
encoded_x = encoder.transform(y)
y= np_utils.to_categorical(encoded_x)
x['x0']=-1
#Separar treino e teste
frame=[x['x0'],x['SepalLengthCm'],x['SepalWidthCm'],x['PetalLengthCm'],x['PetalWidthCm']]
x=pd.DataFrame(frame).T
y=pd.DataFrame(y)
X_train=pd.concat([x.iloc[:25,:],x.iloc[50:75,:],x.iloc[100:125,:]])
x_test=pd.concat([x.iloc[25:50,:],x.iloc[75:100,:],x.iloc[125:,:]])
y_train=pd.concat([y.iloc[:25,:],y.iloc[50:75,:],y.iloc[100:125,:]])
y_test=pd.concat([y.iloc[25:50,:],y.iloc[75:100,:],y.iloc[125:,:]])
def saida_neuronio(vetor_peso, vetor_exemplo):
    v=0
    
    for i in range(len(vetor_exemplo)):
        v=v+vetor_peso[i]*vetor_exemplo[i]
        
    if(v>0):
        return 1
    else:
        return 0
def treinamento(erro_max,x,y,eta,w):
    erro_total=erro_max
    epoca=0
    atualizacao=0
    w=[0,0,0,0,0]
    lista_erro=[]
    while(erro_total>=erro_max):
        epoca+=1
        #print('#EPOCA:'+str(epoca))
        
        for i in range(len(x)):

                f=saida_neuronio(w,x[i])
                delta=(y[i][0]-f)
                erro=delta*delta
                erro_total=erro_total+erro
                #print("y["+str(i)+"]=" + str(y[i][0])+ ",f="+str(f))
                
                if(erro>0):
                    atualizacao=atualizacao+1
                    for j in range(len(w)):
                        w[j]=w[j] + eta * x[i][j] * delta
                        
                        #print(", w["+str(j)+"]=" + str(w[j]))
                    #print('Atualização='+ str(atualizacao) + '\n')
       
        erro_total=0
        #print()
        for i in range(len(x)):

                f=saida_neuronio(w,x[i])
                delta=(y[i][0]-f)
                erro=delta*delta
                erro_total=erro_total+erro
        #print("erro_total="+str(erro_total)+ '\n')
        lista_erro.append(erro_total)  
                
    return w,lista_erro
       
w=[0,0,0,0,0]                    
vetor_w,lista_erro=treinamento(0.001,X_train.iloc[:,:].values,y_train.iloc[:,:].values,1,w)
print("Vetor de peso: "+ str(vetor_w))
def avaliar_setosa(x,y,w):
    erro_total=0
    for i in range(len(x)):

        f=saida_neuronio(w,x[i])
        delta=(y[i][0]-f)
        erro=delta*delta
        erro_total=erro_total+erro
    return erro_total
#Calcular o erro no conjunto de teste
avaliar_setosa(x_test.iloc[:,:].values,y_test.iloc[:,:].values,vetor_w)

def treinamento_virginica(erro_max,x,y,eta,num_epoca,w):
    erro_total=erro_max
    epoca=0
    atualizacao=0
    lista_erro=[]
    while(num_epoca>0):
        epoca+=1
        #print('#EPOCA:'+str(epoca))
        
        for i in range(len(x)):

                f=saida_neuronio(w,x[i])
                delta=(y[i][2]-f)
                erro=delta*delta
                erro_total=erro_total+erro
                #print("y["+str(i)+"]=" + str(y[i][2])+ ",f="+str(f))
                
                if(erro>0):
                    atualizacao=atualizacao+1
                    for j in range(len(w)):
                        w[j]=w[j] + eta * x[i][j] * delta
                        
                        #print(", w["+str(j)+"]=" + str(w[j]))
                    #print('Atualização='+ str(atualizacao) + '\n')
                
        erro_total=0
        num_epoca=num_epoca-1
        #print()
        for i in range(len(x)):

                f=saida_neuronio(w,x[i])
                delta=(y[i][2]-f)
                erro=delta*delta
                erro_total=erro_total+erro
        #print("erro_total="+str(erro_total)+ '\n')
        lista_erro.append(erro_total)
                
    return w,lista_erro
       
w=[0,0,0,0,0]                 
vetor_w,lista_erro=treinamento_virginica(0.001,X_train.iloc[:,:].values,y_train.iloc[:,:].values,1,100,w)
print("Vetor de peso: "+ str(vetor_w))
def avaliar_virginica(x,y,w):
    erro_total=0
    for i in range(len(x)):

        f=saida_neuronio(w,x[i])
        delta=(y[i][2]-f)
        erro=delta*delta
        erro_total=erro_total+erro
    return erro_total
#Calcular o erro no conjunto de teste
avaliar_virginica(x_test.iloc[:,:].values,y_test.iloc[:,:].values,vetor_w)
def repetir_treinamento():
    
    data=pd.DataFrame(index=range(1))
    data['mediaA_taxa0.1']=0
    data['mediaB_taxa0.1']=0
    data['desvioA_taxa0.1']=0
    data['desvioB_taxa0.1']=0
        
    data['mediaA_taxa1']=0
    data['mediaB_taxa1']=0
    data['desvioA_taxa1']=0
    data['desvioB_taxa1']=0
    
    data['mediaA_taxa10']=0
    data['mediaB_taxa10']=0
    data['desvioA_taxa10']=0
    data['desvioB_taxa10']=0 
    
    L1_1=[]
    L1_2=[]
    
    
    L2_1=[]
    L2_2=[]
   
    
    L3_1=[]
    L3_2=[]
   
    
    for i in range(30):
        w=np.random.rand(5)
        #aprendizagem 0.1
        t2_1,lista_errot2_1=treinamento_virginica(0.001,X_train.iloc[:,:].values,y_train.iloc[:,:].values,0.1,100,w)
        t1_1,lista_errot1_1=treinamento(0.001,X_train.iloc[:,:].values,y_train.iloc[:,:].values,0.1,w)
        V2=avaliar_virginica(x_test.iloc[:,:].values,y_test.iloc[:,:].values,t2_1)
        V1=avaliar_setosa(x_test.iloc[:,:].values,y_test.iloc[:,:].values,t1_1)
        L1_1.append(V1)
        L1_2.append(V2)
        
        #aprendizagem 1
        t2_2,lista_errot2_2=treinamento_virginica(0.001,X_train.iloc[:,:].values,y_train.iloc[:,:].values,1,100,w)
        t1_2,lista_errot1_2=treinamento(0.001,X_train.iloc[:,:].values,y_train.iloc[:,:].values,1,w)
        V2=avaliar_virginica(x_test.iloc[:,:].values,y_test.iloc[:,:].values,t2_2)
        V1=avaliar_setosa(x_test.iloc[:,:].values,y_test.iloc[:,:].values,t1_2)
        L2_1.append(V1)
        L2_2.append(V2)
        
        
        #aprendizagem 10
        t2_3,lista_errot2_3=treinamento_virginica(0.001,X_train.iloc[:,:].values,y_train.iloc[:,:].values,10,100,w)
        t1_3,lista_errot1_3=treinamento(0.001,X_train.iloc[:,:].values,y_train.iloc[:,:].values,10,w)
        V2=avaliar_virginica(x_test.iloc[:,:].values,y_test.iloc[:,:].values,t2_3)
        V1=avaliar_setosa(x_test.iloc[:,:].values,y_test.iloc[:,:].values,t1_3)
        L3_1.append(V1)
        L3_2.append(V2)
        
        
    data['mediaA_taxa0.1']=statistics.mean(L1_1)
    data['desvioA_taxa0.1']=statistics.stdev(L1_1)
    data['mediaB_taxa0.1']=statistics.mean(L1_2)
    data['desvioB_taxa0.1']=statistics.stdev(L1_2)
    
    data['mediaA_taxa1']=statistics.mean(L2_1)
    data['desvioA_taxa1']=statistics.stdev(L2_1)
    data['mediaB_taxa1']=statistics.mean(L2_2)
    data['desvioB_taxa1']=statistics.stdev(L2_2)
    
    data['mediaA_taxa10']=statistics.mean(L3_1)
    data['desvioA_taxa10']=statistics.stdev(L3_1)
    data['mediaB_taxa10']=statistics.mean(L3_2)
    data['desvioB_taxa10']=statistics.stdev(L3_2)
    
    return data
        
data=repetir_treinamento()    
data