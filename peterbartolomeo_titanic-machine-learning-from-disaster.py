import pandas as pd #Esta libreria se usó para leer los datos csv y guardar los resultados
import numpy as np  #Con numpy se castearon los numeros flotantes de las predicciones a enteros
from sklearn.linear_model import LinearRegression # Libreria para realizar regresiones lineales vista en clase
# Lectura de datos con pandas version web con los datos de kaggle 
entrenamiento = pd.read_csv('../input/train.csv')
prueba = pd.read_csv('../input/test.csv')
# Drop quita las columnas que no se van a ocupar y se crean variables dummies para que cuadre la información
entrenamiento = entrenamiento.drop(columns = ['PassengerId','Name','Ticket','Cabin'], axis = 1)
prueba = prueba.drop(columns = ['PassengerId','Name','Ticket','Cabin'], axis = 1)
entrenamiento = pd.get_dummies(entrenamiento, columns = ['Sex','Embarked'])
prueba = pd.get_dummies(prueba, columns = ['Sex','Embarked'])
#Los datos sin valor se setean en ceros
entrenamiento.fillna(0, inplace=True)
prueba.fillna(0, inplace=True)

# y es la variable que se va a predecir y X es la informacion que se ocupa para calcular Y
y = entrenamiento['Survived'].values
entrenamiento = entrenamiento.drop(columns = ['Survived'], axis = 1)
X = entrenamiento.values
# se verifica que la información concuerde imprimiendo la estructura de X Y
print(X.shape)
print(y.shape)
print(prueba.values.shape)
# se aplica la regresión lineal para hacer las predicciones de los datos de prueba
lin_reg = LinearRegression()
lin_reg.fit(X,y)
predicciones = lin_reg.predict(prueba.values)
# el metodo predict devuelve valores flotante por lo que se realiza un cast de flotantes a enteros
predicciones = np.rint(predicciones)
print(predicciones.shape)
# se vulelve a obtener la informacion del archivo csv para recuperar la columna dropeada 'PassengerId'
archivo = pd.read_csv('../input/test.csv')

# se le agregan las predicciones a la información
archivo['Survived'] = predicciones
# se mantienen solo las columnas consideradas para la evaluacion en linea
archivo = archivo[['PassengerId','Survived']]

# finalmente se exporta el archivo
archivo.to_csv('archivo.csv', sep=',', encoding='utf-8')
#se imprime el archivo
print(archivo)