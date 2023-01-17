# Import libraries
import pandas as pd
import sklearn
# Databases Read
cr_test = pd.read_csv("../input/test.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="NaN")
cr_train = pd.read_csv("../input/train.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                      na_values="NaN")

#Tamanho base de teste
cr_test.shape
# Tamanho base de treino
cr_train.shape
#import biblioteca de plot
import matplotlib.pyplot as plt
# Plot 1: owns a tablet (no = 0, yes = 1)
cr_train["v18q"].value_counts().plot(kind = "bar")
# Plot 2: Total persons in the household
cr_train["r4t3"].value_counts().plot(kind = "bar")
# Plot 3: Years of Schooling
cr_train["escolari"].value_counts().plot(kind = "bar")
# Plot 4: Number of children 0 to 19 in household
cr_train["hogar_nin"].value_counts().plot(kind = "bar")
# Plot 5: Target
cr_train["Target"].value_counts().plot(kind = "bar")
# Plot 6: Scholarity (Target = 1)
data_t1 = cr_train[cr_train.Target == 1] 
data_t1["escolari"].value_counts().plot(kind = "bar")
# Substituindo NaN pela média da coluna
means = cr_train.mean()
cr_train = cr_train.fillna(means)
# Convertendo valores em texto para números 
from sklearn import preprocessing
cr_train_num = cr_train.iloc[:,0:142].apply(preprocessing.LabelEncoder().fit_transform)
cr_test_num = cr_test.iloc[:,0:142].apply(preprocessing.LabelEncoder().fit_transform)
# Selecionando features
# v2a1, Monthly rent payment
# rooms,  number of all rooms in the house
# v14a, =1 has bathroom in the household
# r4t3, Total persons in the household
# tamviv, number of persons living in the household
# escolari, years of schooling
# hogar_nin, Number of children 0 to 19 in household
# hogar_total, # of total individuals in the household
# dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
# instlevel1, =1 no level of education 
# instlevel8, =1 undergraduate and higher education
# tipovivi4, =1 precarious
# area1, =1 zona urbana
# area2, =2 zona rural


cr_test_X = cr_test_num[["v2a1","rooms","v14a","r4t3","tamviv","escolari","hogar_nin","hogar_total","dependency","instlevel1","instlevel8","tipovivi4","area1","area2"]]
cr_train_X = cr_train_num[["v2a1","rooms","v14a","r4t3","tamviv","escolari","hogar_nin","hogar_total","dependency","instlevel1","instlevel8","tipovivi4","area1","area2"]]
# Selecionando Target da Base de treino
cr_train_Y = cr_train.Target
## kNN
# Import biblioteca para utilizar classificador kNN
from sklearn.neighbors import KNeighborsClassifier
# Definição do classificador kNN
knn = KNeighborsClassifier(n_neighbors=30)
# Fit no kNN
knn.fit(cr_train_X,cr_train_Y)
# Import bibliteca para realização de "Cross Validation"
from sklearn.model_selection import cross_val_score
# Import numpy
import numpy as np
#Realização de "Cross Validation" melhor resultado de cv
scores = cross_val_score(knn, cr_train_X, cr_train_Y, cv = 525)
# Media dos scores de cada bloco de dados
print(np.mean(scores))
# Predict
cr_test_Y = knn.predict(cr_test_X)
cr_test_Y
# Editing prediction
submission_id =  cr_test.Id
submission_pred = cr_test_Y
sub = pd.DataFrame({'Id':submission_id[:],'Target':submission_pred[:]})
sub.Target.value_counts()
