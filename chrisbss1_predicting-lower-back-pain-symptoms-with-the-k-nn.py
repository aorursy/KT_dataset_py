import pandas as pd



data = pd.read_csv('../input/lower-back-pain-symptoms-dataset/Dataset_spine.csv')

data.head()
data = data[[i for i in data.columns if i not in ['Unnamed: 13'] ]]

data.head()
data.Class_att.replace(['Abnormal', 'Normal'], [1, 0], inplace=True)

data.head()
data = data.rename(columns = {'Col1':'pelvic_incidence','Col2':'pelvic_tilt', 'Col3':'lumbar_lordosis_angle', 'Col4':'sacral_slope', 'Col5':'pelvic_radius', 'Col6':'degree_spondylolisthesis', 'Col7':'pelvic_slope', 'Col8':'Direct_tilt', 'Col9':'thoracic_slope', 'Col10':'cervical_tilt', 'Col11':'sacrum_angle', 'Col12':'scoliosis_slope', 'Class_att':'Pain'})

data.head(5)
import matplotlib.pyplot as plt



abnormal= len(data[data['Pain']==0])

normal= len(data[data['Pain']==1])



y= [abnormal,normal]

x= ['Abnormal','Normal']



plt.xlabel('')

plt.ylabel('Number of people')

plt.bar(x,y, color ='sandybrown',width=0.20)

plt.show()
from sklearn.model_selection import train_test_split

y = 'Pain'

X = [c for c in list(data) if c not in [y]]

X_data = data[X]

y_data = data[[y]]



xtrain, xtest, ytrain, ytest = train_test_split(X_data, y_data, train_size=0.7, random_state=1)
from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from sklearn.neighbors import KNeighborsClassifier

import numpy as np



test = []

train=[]





x = list(np.arange(3,70,1))





for i in x :

  model= KNeighborsClassifier(n_neighbors=i,metric ='minkowski', p = 1, algorithm ='kd_tree', n_jobs =-1)

  model.fit(xtrain, ytrain)

  fpr, tpr, _ = metrics.roc_curve(np.array(ytest), model.predict_proba(xtest)[:,1])

  auc_test = metrics.auc(fpr,tpr)

  fpr, tpr, _ = metrics.roc_curve(np.array(ytrain), model.predict_proba(xtrain)[:,1])

  auc_train = metrics.auc(fpr,tpr)

  test.append(auc_test)

  train.append(auc_train)







best_value = x[test.index(max(test))]



plt.plot(x,train, color="orange",  label='Training')

plt.plot(x,test, color="green",  label='Test') 







plt.xlabel('Number of neighbors')

plt.ylabel('AUC')

plt.legend()





plt.show()



print()

print('The optimal value is ' + str(best_value))
model= KNeighborsClassifier(n_neighbors=best_value,metric ='minkowski', p = 1, algorithm ='kd_tree', n_jobs =-1)

model.fit(xtrain, ytrain)



# AUC

fpr, tpr, _ = metrics.roc_curve(np.array(ytest), model.predict_proba(xtest)[:,1])

auc = metrics.auc(fpr,tpr)



# Accuracy



accuracy = model.score(xtest, ytest)



print('Number of neighbors : ' + str(best_value))

print()

print('AUC : ' + str(auc))

print()

print('Accuracy : ' + str(accuracy))