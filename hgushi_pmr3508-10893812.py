import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
train = pd.read_csv("../input/dataset/train_data.csv",
                   na_values = '?')
train = train.dropna()
Atrain = train[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Btrain = train.income
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=15)
cross_val_score(knn, Atrain, Btrain, cv=10)
knn.fit(Atrain,Btrain)
test = pd.read_csv("../input/dataset/test_data.csv",
                   na_values = '?')
Atest = test[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Bpred=knn.predict(Atest)
Bpred
prediction = pd.DataFrame(index = test.index)
prediction["Id"]=test.Id
prediction['income'] = Bpred
prediction
prediction.to_csv("submition.csv",index=False)