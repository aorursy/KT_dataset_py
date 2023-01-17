import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, cross_val_score



%matplotlib inline
input_data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

input_data.head()
input_data.describe()
label_map = {

    'M': 1,

    'B': 0

}



input_data['diagnosis'] = input_data['diagnosis'].map(label_map)



Y = input_data.diagnosis

X = input_data.drop(['Unnamed: 32','id','diagnosis'], axis = 1)

X.head()
ax = sns.countplot(Y ,label="Count") 

B, M = Y.value_counts()



print('Benign: ',B)

print('Malignant : ',M)
sns.set(style="whitegrid", palette="muted")



output_d = Y

input_d = X

input_d = (input_d - input_d.mean()) / (input_d.std())              

input_d = pd.concat([output_d,input_d.iloc[:,0:20]],axis=1)

input_d = pd.melt(input_d,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(20,10))

sns.swarmplot(x="features", y="value", hue="diagnosis", data=input_d)



plt.xticks(rotation=90)
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
drop_list1 = ['perimeter_mean','radius_mean','compactness_mean',

              'concave points_mean','radius_se','perimeter_se',

              'radius_worst','perimeter_worst','compactness_worst',

              'concave points_worst','compactness_se',

              'concave points_se','texture_worst','area_worst']



mod_X = X.drop(drop_list1, axis = 1)    



print("Original Data ::")

print("Samples: {}, Features: {}\n".format(X.shape[0],X.shape[1]))



print("Data after feature selection ::")

print("Samples: {}, Features: {}\n".format(mod_X.shape[0],mod_X.shape[1]))



mod_X.head()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(mod_X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
mod_X = StandardScaler().fit(mod_X).transform(mod_X.astype(float))

mod_X[0]
x_train, x_test, y_train, y_test = train_test_split(mod_X, Y, test_size=0.2)



print ('Train set:', x_train.shape,  y_train.shape)

print ('Test set:', x_test.shape,  y_test.shape)
k = [x for x in range(2,11)]

train_scores = []



for i in k:

    knn = KNeighborsClassifier(n_neighbors = i)

    #print(knn)

    history = knn.fit(x_train,y_train)

    train_acc = accuracy_score(y_train, knn.predict(x_train))

    train_scores.append(train_acc)

    

for i in k:

    print("For K = {}, acc = {}".format(i, train_scores[i-2]))

    

print("\nMaximum Accuracy {:.3f} when K = {}".format(max(train_scores), train_scores.index(max(train_scores))+2))
knn = KNeighborsClassifier(n_neighbors = 3)

history = knn.fit(x_train,y_train)
preds = knn.predict(x_test)



print("Train set Accuracy: ", accuracy_score(y_train, knn.predict(x_train)))



print("\nTest set Accuracy: ", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print("Confusion Matrix: \n",confusion_matrix(y_test, preds))



print("\nROC AUC Score: {:.3f}".format(roc_auc_score(y_test, preds)))