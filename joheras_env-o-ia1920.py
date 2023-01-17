import numpy as np

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
traindf = pd.read_csv("/kaggle/input/ia1920/train.csv")
traindata = traindf.values
descriptores,clases = traindata[:,0:-1],traindata[:,-1]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(descriptores,clases)
testdf = pd.read_csv("/kaggle/input/ia1920/test.csv")

testdata = testdf.values
predicciones = knn.predict(testdata)
soldf = pd.DataFrame([(i+1,int(pred)) for i, pred in enumerate(predicciones)])
soldf.columns = ['Id','Prediction']
soldf
soldf.to_csv("submission.csv",sep=',',index=False)