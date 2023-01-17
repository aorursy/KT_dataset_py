import pandas as pd

from sklearn import datasets

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split



#Modelos 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import  classification_report 

from sklearn.neighbors import KNeighborsClassifier

#Validação 

from sklearn.model_selection import cross_val_score

#Otimizar hiperarametros 

from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()

ver = pd.DataFrame(iris.data)

ver.head

ver.info()
#Separando dados de trino e teste



x = iris.data[:, :2] # pegamos as primeiras duas features aqui. Pegue as outras se quiser.

y = iris.target #classes de cada elemento 

y[0] # classe do primeiro elemento

x[:1, :] # primeiro elemento a classificar



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state = 42 )

#treinamento 

rfc = RandomForestClassifier() 

rfc.fit(x_train, y_train)

rfc_y = rfc.predict(x_test) #teste



# Metricas de qualidade 

print(classification_report(y_test,rfc_y))

#treinamento  usando  otimização de parametros

rfc = RandomForestClassifier(n_estimators= 300 ) 

rfc.fit(x_train, y_train)

rfc_y = rfc.predict(x_test) #teste



# Metricas de qualidade 

print(classification_report(y_test,rfc_y))

#treinando

knn = KNeighborsClassifier(n_neighbors= 3)

knn.fit(x_train, y_train)

knn_y = knn.predict(x_test)





#Metricas

print(classification_report(y_test, knn_y))
#treinando usando  otimização de parametros

knn = KNeighborsClassifier(n_neighbors= 20)

knn.fit(x_train, y_train)

knn_y = knn.predict(x_test)





#Metricas

print(classification_report(y_test, knn_y))
cv_rfc = cross_val_score(rfc,x,y)

cv_knn = cross_val_score(knn,x,y)

print('\nValidação cruzada: {0} vs {1}\n'.format( cv_rfc,cv_knn))
# Achando o desempenho médio do Random Forest

sum_cv_rfc = 0

for cv_score in cv_rfc:

    sum_cv_rfc+= cv_score 

print('\nResultado Random Forest: {0}'.format( sum_cv_rfc/5))    

    

# Achando o desempenho médio do KNN

sum_cv_knn = 0

for cv_score in cv_knn:

    sum_cv_knn+= cv_score 

print('\nResultado KNN: {0}'.format( sum_cv_knn/5))
#Random Forest



parameters_rfc = {'n_estimators': [5,300]}

rfc_hps =  GridSearchCV(rfc, parameters_rfc)

rfc_hps.fit(x,y)

print('Melhor valor para n_estimators: {0}'.format(rfc_hps.best_params_['n_estimators'])) 
#KNN



parameters_knn = {'n_neighbors': (1,20)}

knn_hps =  GridSearchCV(knn, parameters_knn)

knn_hps.fit(x,y)

print('Melhor valor para n_neighbors: {0}'.format(knn_hps.best_params_['n_neighbors'])) 