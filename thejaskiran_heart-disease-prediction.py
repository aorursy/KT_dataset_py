import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/heart-disease-dataset/heart.csv')
data.head()
data.describe()
sns.heatmap(data.isna())
from sklearn.model_selection import train_test_split
x = data.drop(['target'],axis = 1)
y = data['target']
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.3)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(xtrain,ytrain)
pred = knn.predict(xtest)
pred
from sklearn.metrics import classification_report, confusion_matrix 
print(classification_report(ytest,pred))
import numpy as np

error_rate = []

for i in range(1,20):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(xtrain,ytrain)

    pred = knn.predict(xtest)

    error_rate.append(np.mean(pred != ytest))

plt.figure(figsize=(10,6))

plt.plot(range(1,20),error_rate,color='blue',ls='--',marker='o')
xtrain
def predict_price(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):    

    x1 = np.zeros(len(x.columns))

    x1[0] = age

    x1[1] = sex

    x1[2] = cp

    x1[3] = trestbps

    x1[4] = chol

    x1[5] = fbs

    x1[6] = restecg

    x1[7] = thalach

    x1[8] = exang

    x1[9] = oldpeak

    x1[10] = slope

    x1[11] = ca

    x1[12] = thal

    return knn.predict([x1])[0]