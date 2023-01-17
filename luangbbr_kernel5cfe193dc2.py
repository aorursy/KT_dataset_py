import sklearn
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from sklearn                        import metrics, svm
from sklearn.linear_model           import LogisticRegression
from sklearn import preprocessing
from sklearn import utils

train_data_r = pd.read_csv("../input/train.csv",sep=",",header=0)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(train_data_r.isnull().sum())


values={'v18q1':0,'v2a1':0 }
train_data_r=train_data_r.fillna(value=values)


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(train_data_r.isnull().sum())
train_data=train_data_r.drop(["Id","v18q","rez_esc","idhogar","r4h1","r4h2","r4h3","r4m1","r4m1","r4m2","r4m3","female","hogar_total","hogar_nin","hogar_adul","hogar_mayor","dependency","edjefe","edjefa","SQBescolari","SQBage","SQBhogar_total","SQBedjefe","SQBhogar_nin","SQBovercrowding","SQBdependency","SQBmeaned","agesq"],axis=1)
train_data=train_data.dropna()

Ytrain = train_data.Target
Xtrain = train_data.drop(["Target"],axis=1)

train_data.head()
#Padronização
"""from sklearn.preprocessing import StandardScaler

values = train_data.values
scaler=StandardScaler()
scaler=scaler.fit_transform(values)
sdt_train_data = pd.DataFrame(scaler)
xtrain=sdt_train_data[:]
ytrain=sdt_train_data.loc[:,115]"""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
lab_enc = preprocessing.LabelEncoder()
ytrain = lab_enc.fit_transform(Ytrain)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
knn.fit(Xtrain,Ytrain)
scores
n_vizinhos=list(range(1,50))
med_scores=[]
for i in n_vizinhos:
    knn = KNeighborsClassifier(n_neighbors=i)
    med_scores.append(cross_val_score(knn, Xtrain, Ytrain, cv=10).mean())

plt.scatter(n_vizinhos,med_scores)
med_scores=np.array(med_scores)
print(med_scores.argmax())
med_scores
knn = KNeighborsClassifier(n_neighbors=scores.argmax())
knn.fit(Xtrain,Ytrain)
teste_data_r = pd.read_csv("../input/test.csv",sep=",",header=0)
teste_data_r=teste_data_r.fillna(0)
teste_data_rd=teste_data_r.drop(["v18q","rez_esc","idhogar","r4h1","r4h2","r4h3","r4m1","r4m1","r4m2","r4m3","female","hogar_total","hogar_nin","hogar_adul","hogar_mayor","dependency","edjefe","edjefa","SQBescolari","SQBage","SQBhogar_total","SQBedjefe","SQBhogar_nin","SQBovercrowding","SQBdependency","SQBmeaned","agesq"],axis=1)
teste_data_rdn=teste_data_rd.dropna()
teste_data=teste_data_rdn.drop(["Id"],axis=1)

arq = open("prediction.csv","w")
Yteste = knn.predict(teste_data)
arq.write("Id,Target\n")
for i,j in zip(teste_data_rdn["Id"],Yteste):
    arq.write(str(i)+","+str(j)+"\n")
    
arq.close()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(teste_data_r.isnull().sum())
