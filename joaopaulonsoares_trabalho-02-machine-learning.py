import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



import h5py

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd 

import os

import numpy as np

from IPython.display import Image

import sys

from sklearn import datasets

from sklearn.model_selection import train_test_split 

from sklearn.neighbors import KNeighborsClassifier 





from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC



from sklearn.metrics import roc_auc_score



file_train = '../input/train.csv'

file_valid = '../input/valid.csv'

file_test = '../input/test.csv'

file_exemplo = '../input/exemplo_resultado.csv'





data_train = pd.read_csv(file_train)

data_valid = pd.read_csv(file_valid)

data_test = pd.read_csv(file_test)





data_exemplo = pd.read_csv(file_exemplo)

print(len(data_train), len(data_valid))#, len(data_test))
Y = data_train['default payment next month']

ID_predict = data_exemplo['ID']



ID_predict_valid = data_valid['ID']

ID_predict_test = data_test['ID']

ID_predict_final = ID_predict_valid.append(ID_predict_test)



data_train = data_train.drop('default payment next month', axis=1)





# Os dados são juntados para um melhor tratamento

data_all = data_train.append(data_valid).append(data_test)

#data_all = data_all.drop('MARRIAGE', axis=1)



len(data_all)


#Overview dos dados

# Número de linhas e colunas

print(data_all.shape, "\n")



# Tipo de dados de cada coluna

print(data_all.dtypes, "\n")



#Descobrir a quantidade de valores nulos

print(data_all.isna().sum(), "\n")
data_all
plt.figure(figsize=(15,15))

ax = plt.axes()

corr = data_all.drop(['ID'], axis=1).corr()

sns.heatmap(corr, vmax=1,vmin=-1, square=True, annot=True, cmap='Spectral',linecolor="white", linewidths=0.01, ax=ax)

ax.set_title('Gráfico de par de coeficiente de correlação',fontweight="bold", size=30)

plt.show()
data_all['PAY_0'] = data_all['PAY_0'].replace(-2, 0)

data_all['PAY_0'] = data_all['PAY_0'].replace(4, 0)



data_all['PAY_2'] = data_all['PAY_2'].replace(-2, 0)

data_all['PAY_2'] = data_all['PAY_2'].replace(4, 0)



data_all['PAY_3'] = data_all['PAY_3'].replace(-2, 0)

data_all['PAY_3'] = data_all['PAY_3'].replace(4, 0)



data_all['PAY_4'] = data_all['PAY_4'].replace(-2, 0)

data_all['PAY_4'] = data_all['PAY_4'].replace(4, 0)



data_all['PAY_5'] = data_all['PAY_5'].replace(-2, 0)

data_all['PAY_5'] = data_all['PAY_5'].replace(4, 0)



data_all['PAY_6'] = data_all['PAY_6'].replace(-2, 0)

data_all['PAY_6'] = data_all['PAY_6'].replace(4, 0)
data_all
from sklearn.model_selection import train_test_split



# Create X

data_train = data_all[:(len(data_train))]

data_valid = data_all[(len(data_train)):((len(data_train))+(len(data_valid)))]

data_test = data_all[((len(data_train))+(len(data_valid))):]



print()



X = data_train

#X = data_test



print(len(data_train), len(data_valid))#, len(data_test))

from sklearn import tree



model_decision_tree = tree.DecisionTreeClassifier()

"""

param_grid = { 

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8,9,10],

    'criterion' :['entropy','gini'],

    'max_leaf_nodes' :[10,100]

}



CV_mdt = GridSearchCV(estimator=model_decision_tree, param_grid=param_grid, cv= 5)

CV_mdt.fit(X, Y)

CV_mdt.best_params_

"""
model_decision_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10,  max_leaf_nodes=100,max_features = 'auto')

model_decision_tree.fit(X, Y)

pred_model_decision_tree= model_decision_tree.predict(X)



print("ROC AUC score:",roc_auc_score(Y, pred_model_decision_tree))

print("Acurácia do Decision Tree Classifier: ", accuracy_score(Y,pred_model_decision_tree) )
rfc = RandomForestClassifier(random_state=42)
"""

param_grid = { 

    'n_estimators': [10,20,30,40,50,500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['entropy']

}



CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

CV_rfc.fit(X, Y)

CV_rfc.best_params_

"""
model_RFC = RandomForestClassifier(criterion='entropy', max_depth=8,  max_features=8, n_estimators = 30)

model_RFC.fit(X, Y)

pred_model_RFC=model_RFC.predict(X)



print("ROC AUC score:",roc_auc_score(Y, pred_model_RFC))



print("Accuracy for Random Forest on CV data: ",accuracy_score(Y,pred_model_RFC))

# 0.8319047619047619
import pandas as pd

feature_importances = pd.DataFrame(model_RFC.feature_importances_,

                                   index = X.columns,

                                    columns=['importance']).sort_values('importance', ascending=False)



feature_importances
from sklearn.neighbors import KNeighborsClassifier 



knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X, Y)
"""

param_grid = { 

    'n_neighbors': [5,6,7],

    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],

    'leaf_size' : [10,30]

}



CV_knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv= 5)

CV_knn.fit(X, Y)

CV_knn.best_params_



# {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 6}

"""


# Ver a acurácia

# Setup arrays to store train and test accuracies

neighbors = np.arange(1, 9)

train_accuracy = np.empty(len(neighbors))

train_auc_roc = np.empty(len(neighbors))



# Loop over different values of k

for i, k in enumerate(neighbors):

    # Setup a k-NN Classifier with k neighbors: knn

    knn = KNeighborsClassifier(n_neighbors=k)



    # Fit the classifier to the training data

    knn.fit(X, Y)

    pred_knn = knn.predict(X)

    #Compute accuracy on the training set

    #train_accuracy[i] = knn.score(X, Y)

    train_auc_roc[i] = roc_auc_score(Y, pred_knn)



# Generate plot

plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, train_auc_roc, label = 'ROC AUC score')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('ROC AUC')

plt.show()

from sklearn.neighbors import KNeighborsClassifier 



knn_p = KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size = 10)

knn_p.fit(X, Y)

knn_model_predict=knn_p.predict(X)



print("ROC AUC score:",roc_auc_score(Y, knn_model_predict))

print("Acurácia do KNN: ",accuracy_score(Y,knn_model_predict))

from sklearn.neighbors import KNeighborsClassifier 



knn_p_2 = KNeighborsClassifier(n_neighbors=6, algorithm='auto', leaf_size = 10)

knn_p_2.fit(X, Y)

knn_model_predict_2=knn_p_2.predict(X)



print("Acurácia do KNN: ",accuracy_score(Y,knn_model_predict_2))
data_predict = data_valid.append(data_test)



Y_predict = model_RFC.predict(data_predict)



Y_predict
print(len(Y_predict))
Default_exemplo = data_exemplo['Default']



#Y_final = np.concatenate((Y_predict, Default_exemplo[len(data_valid):]), axis=0)

ID_predict_final = ID_predict_valid.append(ID_predict_test)





print(len(Y_predict), len(ID_predict_final))


data_to_submit = pd.DataFrame({

    'ID': ID_predict_final,

    'Default':Y_predict

})



data_to_submit.to_csv('csv_to_submit.csv', index = False)