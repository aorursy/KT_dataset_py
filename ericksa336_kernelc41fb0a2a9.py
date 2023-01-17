import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets, svm, metrics #para crear la matriz
% matplotlib inline

# Cargamos el dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

X_train = (train.iloc[:,1:].values).astype('float32') # todos los valores
Y_train = train.iloc[:,0].values.astype('int32') # los label y digitos
X_test = test.values.astype('float32')

#Se sabe que las imágenes no son en blanco y negro sino que están en escala de grises, por lo tanto se realiza la normalización para reducir el efecto de las diferencias de iluminación. Todos aquellos valores de columnas que superen el valor 0 se convierten a 1.



X_test[X_test>0]=1
X_train[X_train>0]=1


train.shape
test.shape
train.head()
test.head()
#utilizamos bayes
from sklearn.naive_bayes import MultinomialNB
clf0 = MultinomialNB()
clf0.fit(X_train, Y_train)
predictions = clf0.predict(X_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("bayes.csv", index=False , header=True)
#creamos la matriz de confusión para bayes
clftest1 = MultinomialNB()
clftest1.fit(X_train, Y_train)
predictions = clftest1.predict(X_train)
submissions=pd.DataFrame({"ImageId": Y_train,
                         "Label": predictions})
#creamos un csv con el digito real vs digito obtenido con predicción
submissions.to_csv("bayestrain.csv", index=False, header=True)
from sklearn import datasets, svm, metrics #para crear la matriz
print("Confusion matrix:\n%s" % metrics.confusion_matrix(Y_train, predictions))
#utilizamos k mas cercanos
from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier(n_neighbors=10)
clf2.fit(X_train, Y_train) 
predictions = clf2.predict(X_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("kcercanos.csv", index=False, header=True)
#creamos la matriz de confusión para k vecinos
clftest2 = KNeighborsClassifier(n_neighbors=10)
clftest2.fit(X_train, Y_train)
predictions = clftest2.predict(X_train)
submissions=pd.DataFrame({"ImageId": Y_train,
                         "Label": predictions})
#creamos un csv con el digito real vs digito obtenido con predicción de kvecinos
submissions.to_csv("kvecinostrain.csv", index=False, header=True)
from sklearn import datasets, svm, metrics #para crear la matriz
print("Confusion matrix:\n%s" % metrics.confusion_matrix(Y_train, predictions))
from sklearn import datasets, svm, metrics #para crear la matriz
print("Confusion matrix:\n%s" % metrics.confusion_matrix(Y_train, predictions))
