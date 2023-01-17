import numpy as np 
import sklearn
from sklearn.datasets import load_iris # traer el dataset de las flores de iris
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
%matplotlib inline
iris = load_iris() # Creamos la variable para traer nuestro dataset
type(iris) # Tipo de dato Bunch es una especie de diccionario 
iris.keys() # Revisamos las llaves de nuestro diccionario
iris['data'] 
iris['target_names']
iris['target']
iris['feature_names'] # Nombre de las mediciones que se le hacen a las flores
X=iris['data'] # Mediciones de las flores
y=iris['target'] # Etiquetas de las flores
# Dividimos nuestros datos en set de entrenamiento y de pruebas 
X_train, X_test, y_train, y_test = train_test_split(X, y,)
X_train.shape # Matriz de 112 flores con 4 mediciones
y_train.shape # Cada una de las etiquetas de las  flores
from sklearn.neighbors import KNeighborsClassifier
KNN_iris=KNeighborsClassifier(n_neighbors=9) # para este caso elegimos k=7
KNN_iris.fit(X_train,y_train) # Recibe como parametros nuestros datos de prueba 
KNN_iris.score(X_test,y_test) # los parametros que recibe son los X and y de testing
KNN_iris.predict([[1.1,1.5,2.5,3.5]]) # Le ingresamos las 4 medidas para que nos diga el tipo
iris.target_names # Lo que quiere decir que pertenece a una versicolor 
pred_iris = KNN_iris.predict(X_test) # Creamos la variable de prediccion
from sklearn.metrics import confusion_matrix,classification_report
# Los parametros que recibe son los y de pruebas y los de prediccion
print(confusion_matrix(y_test,pred_iris)) 
print('\n')
print(classification_report(y_test,pred_iris))


# Creamos un arreglo vacio para llenarlo con los valores a medida que cambie los puntos vecinos

error_rate = [] 
# Will take some time
for i in range(1,60):
    
    KNN_iris = KNeighborsClassifier(n_neighbors=i) # Instancia con numeros vecinos 
    KNN_iris.fit(X_train,y_train)
    pred_i = KNN_iris.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test)) # Zona de carga del vector de error donde calcula el promedio
                                                 # de los valores que son diferentes a los valores reales
# Creamos nuestro grafico y lo personalizamos
plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

