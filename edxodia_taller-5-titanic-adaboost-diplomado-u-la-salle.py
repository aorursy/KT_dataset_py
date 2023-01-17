import pandas as pd              # pandas es ideal para administrar conjuntos de datos como un solo objeto

import numpy as np               # numpy tiene excelentes operaciones de matriz / matemáticas en matrices



import matplotlib.pyplot as plt  # figuras y gráficos

import seaborn as sns            # un complemento elegante para matplotlib



# Este comando crea las figuras y gráficos dentro del notebook Jupyter

%matplotlib inline     
'''

Esto llama a todas las funciones de preprocesamiento.

El preprocesamiento debe hacerse exactamente igual en los conjuntos de datos de entrenamiento y prueba.

'''



def preprocessData(df):

    

    # Convierta el campo de embarked de categórico ("S", "C", "Q")

    # para numerico (0,1,2)

    df = convertEmbarked(df)

    

    # Convierta sex. Female = 0, Male = 1

    df = convertSex(df)



    df = addFamilyFactor(df)

    

    df = addTitles(df)



    # Remover columnas irrelevantes y no numericas (features)

    df = df.drop(['Name', 'Cabin', 'PassengerId', 'Ticket'], axis=1) 



    # Reemplazar valores faltantes (NaN) with the mean value for that field

    df = replaceWithMean(df)



    return df
'''

convierte el campo sex a numerico

'''

def convertSex(df):

    

    

    # Cree una nueva columna llamada 'Gender' que es un map de la columna "Sex" en valores enteros 

    '''escriba su codigo'''

    # Ahora elimine la columna "sex" ya que la hemos reemplazado por la columna 'gender'

    '''escriba su codigo'''

    

    return df

    
'''

Scikit-learn solo puede manejar números.

Así que reemplacemos los valores de texto para la columna 'Embarked' con números. 

Por ejemplo, el puerto de embarque etiquetado 'S' es 0, 'C' es 1, y 'Q' es 2.

'''

def convertEmbarked(df):

    

    if ('Embarked' in df.columns) :  # 

        

        # valor faltante, llene na con el valor más frecuente 

        if (len(df[df["Embarked"].isnull()]) > 0):



            # Necesitamos deshacernos de los valores faltantes

            # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mode.html

            # Si desea imputar valores faltantes con la moda en un dataframe de datos df, puede hacerlo:

            df.loc[df["Embarked"].isnull(), 'Embarked'] = df["Embarked"].dropna().mode().iloc[0]



        ports = list(enumerate(np.unique(df["Embarked"])))  # Obtenemos la lista ID de puertos únicos

        port_dict = { name: i for i, name in ports } # Cree un diccionario de las diferentes ID de puerto

        df["Embarked"] = df["Embarked"].map( lambda x: port_dict[x]).astype(int)  # Reasignar los ID de puerto a números

        

    return df
def addTitles(df):

    

    # extraemos el título de cada nombre 

    combined = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # construimos un mapa de títulos 

    Title_Dictionary = {

                        "Capt":       1,

                        "Col":        1,

                        "Major":      1,

                        "Jonkheer":   3,

                        "Don":        3,

                        "Sir" :       3,

                        "Dr":         2,

                        "Rev":        1,

                        "the Countess":3,

                        "Dona":       3,

                        "Mme":        0,

                        "Mlle":       0,

                        "Ms":         0,

                        "Mr" :        0,

                        "Mrs" :       0,

                        "Miss" :      0,

                        "Master" :    1,

                        "Lady" :      3



                        }

    

    # mapeamos cada título

    df['Title'] = combined.map(Title_Dictionary)

    

    return df
'''

Reemplace los valores faltantes (NaN) con el valor medio para esa columna

'''

def replaceWithMean(df):

    

    '''escriba su codigo'''

    return df
'''

existen 2 variables de tipo "family" en el dataset. podemos combinarlas en una variable.

'''

def addFamilyFactor(df):

    # Agregar una categoría llamada FamilyFactor

    # ¿Quizás las personas con familias más grandes tenían una mayor probabilidad de rescate?

    # Si solo agrego los dos juntos, entonces la nueva categoría es solo una transformación lineal y

    # realmente no agregará nueva información. Entonces agrego y luego elevo al cuadrado el valor. 

    '''escriba su codigo'''

    

    return df
'''



Lea los datos de entrenamiento del archivo csv path='../input/train.csv'



'''

train_df = '''escriba su codigo'''



# Obtenga la información básica para los datos en este archivo

train_df.info()
# Configure un grafico con 3 subgraficos una al lado de la otra 

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



# Cuente cuántas personas se embarcaron en cada ubicación

# countplot es para categorial data, barplot para data cuantitativa 

sns.countplot(x="Embarked", data=train_df, ax=axis1)

axis1.set_title("# pasajeros por sitio de embarque")



# Comparación de sobrevivientes versus muertes en función del embarque 

'''escriba su codigo'''

axis2.set_title("Sobrevivientes versus muertes")



# agrupar por embarked, y obtenga la media de los pasajeros sobrevivientes para cada valor en  Embarked

embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

axis3.set_title("Supervivencia versus embarque")
train_df = preprocessData(train_df)
train_df.describe()
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.svm import SVC



# Los datos ya están listos para funcionar. Así que vamos al entrenamiento

# 

train_data = train_df.values

train_features = train_data[0::,1::]   # los features a usar por el modelo de prediccion (e.g. age, family size)

train_result = train_data[0::,0]       # Lo que predice el modelo (i.e. survived)



print('Entrenamiento con el modelo. Por favor espera ...')



# Adaboost usando un montón de modelos RandomForest

#  

# Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.

# para mas informacion, http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html

model = AdaBoostClassifier(RandomForestClassifier(n_estimators = 1000),

                         algorithm="SAMME",

                         n_estimators=500)





# Ajustar los datos de entrenamiento al modelo Adaboost

model = model.fit(train_features, train_result)



print ('Okay. Terminé de entrenar al modelo.')
from sklearn.metrics import accuracy_score



print ('Accuracy = {:0.2f}%'.format(100.0 * accuracy_score(train_result, model.predict(train_features))))
from sklearn.model_selection import cross_val_score



# Cálculo de cross validation del modelo de entrenamiento

print ('Cálculo de cross validation del modelo de entrenamiento. Por favor espera ...')



# Cross-validation con k-fold de 5. Por lo tanto, esto dividirá aleatoriamente los datos de entrenamiento en dos conjuntos.

# Luego ajusta un modelo a un conjunto y lo prueba contra el otro para obtener una precisión.

# Lo hará 5 veces y devolverá la precisión promedio.

scores = cross_val_score(model, train_features, train_result, cv=5)

print ( 'En promedio, este modelo es correcto {:0.2f}% (+/- {:0.2f}%) .'.format(

        scores.mean() * 100.0, scores.std() * 2 * 100.0))
# Importe los datos de test en un dataframe Pandas 

'''escriba su codigo'''     # cargue los datos de test usando el path '../input/test.csv'



test_df.info()
# obtiene primero los ID de los pasajeros, ya que son eliminados por la función de preprocesamiento

testIds = test_df['PassengerId']



test_df = preprocessData(test_df)
print('Predecir la supervivencia a partir de los datos de test. POR FAVOR ESPERA... ', end='') 



test_predictions = model.predict(test_df.values)



print('FIN')