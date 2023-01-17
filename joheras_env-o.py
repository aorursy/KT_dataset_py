import numpy as np

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
traindf = pd.read_csv("../input/train.csv")
traindf.drop(['Gender'], axis=1, inplace=True) 
traindata = traindf.values
descriptores,clases = traindata[:,0:-1],traindata[:,-1]
clases = [1 if clase==True else 0 for clase in clases]
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(descriptores,clases)
testdf = pd.read_csv("../input/test.csv")

testdf.drop(['Gender'], axis=1, inplace=True) 

testdata = testdf.values
predicciones = knn.predict(testdata)
predicciones = [True if prediccion==1 else False for prediccion in predicciones]
soldf = pd.DataFrame(list(enumerate(predicciones)))
soldf.columns = ['Id','Prediction']
soldf
soldf.to_csv("submission.csv",sep=',',index=False)