import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.cluster import KMeans

from sklearn.metrics import pairwise_distances_argmin_min



%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')
df4=pd.read_csv("../input/heart.csv",decimal=',')
df4.tail(2)
df4.describe()
# Vemos la relación entre o y 1

print(df4.groupby('target').size())
# Analizamos las variables a reducir

sb.pairplot(df4.dropna(), hue='target',height=5,vars=["age","trestbps","chol","thalach","oldpeak"],kind='scatter')
#Definimos las entradas

X = np.array(df4[["age","trestbps","chol","thalach","oldpeak"]])

y = np.array(df4['target'])

X.shape
Nc = range(1, 10)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

score

plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
# Probamos con K=2

kmeans = KMeans(n_clusters=2).fit(X)

centroids = kmeans.cluster_centers_

print(centroids)

#llevamos los datos ordenados a una variable

labels=kmeans.predict(X)

print(labels)
labels.dtype
#vemos el representante del grupo, el enfermo cercano a su centroid

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

closest
#Clasificar nuevos pacientes

X_new = np.array([[57,130,236,174,0]]) # paciente 1



new_labels = kmeans.predict(X_new)

print(new_labels)
#Añadimos los datos ordenados al dataframe(df4)

df4['kmean']=labels
df4.head(1)
#Borramos las columnas que sobran, ahora están incluidas en kmean

drop_elements = ['age','trestbps','chol','thalach','oldpeak']

df5 = df4.drop(drop_elements, axis = 1)
df5.tail(1)

         
#Reordenamos las columnas

df6=df5[ ['target','sex', 'cp', 'fbs','restecg','exang','slope','ca','thal','kmean'] ]

df6.dropna(inplace=True)

df6.head()
# Imports needed for the script

import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont
# Probamos el algoritmo y buscamos el mejor resultado

cv = KFold(n_splits=20) # Numero deseado de "folds" que haremos

accuracies = list()

max_attributes = len(list(df5))

depth_range = range(1, max_attributes + 1)



# Testearemos la profundidad de 1 a cantidad de atributos +1

for depth in depth_range:

    fold_accuracy = [] 

    tree_model = tree.DecisionTreeClassifier(criterion='entropy',

                                             min_samples_split=10,

                                             min_samples_leaf=3,

                                             max_depth = depth,

                                             class_weight={1:1.24})

    for train_fold, valid_fold in cv.split(df5):

        f_train = df5.loc[train_fold] 

        f_valid = df5.loc[valid_fold] 



        model = tree_model.fit(X = f_train.drop(['target'], axis=1), 

                               y = f_train["target"]) 

        valid_acc = model.score(X = f_valid.drop(['target'], axis=1), 

                                y = f_valid["target"]) # calculamos la precision con el segmento de validacion

        fold_accuracy.append(valid_acc)



    avg = sum(fold_accuracy)/len(fold_accuracy)

    accuracies.append(avg)

    

# Mostramos los resultados obtenidos

df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})

df = df[["Max Depth", "Average Accuracy"]]

print(df.to_string(index=False))
# Crear arrays de entrenamiento y las etiquetas que indican si tieno o no la enfermedad

y_train = df5['target']

x_train = df5.drop(['target'], axis=1).values 



# Crear Arbol de decision con profundidad = 5 (Donde obtuvimos un 0.824583 )

decision_tree = tree.DecisionTreeClassifier(criterion='entropy',

                                            min_samples_split=10,

                                            min_samples_leaf=3,

                                            max_depth = 5,

                                            class_weight={1:1.24})# Obtenido como relación entre 0 y 1 en target

decision_tree.fit(x_train, y_train)



# exportar el modelo a archivo .dot

with open(r"tree2.dot", 'w') as f:

     f = tree.export_graphviz(decision_tree,

                              out_file=f,

                              max_depth = 5,

                              impurity = True,

                              feature_names = list(df5.drop(['target'], axis=1)),

                              class_names = ['No', 'Illness'],

                              rounded = True,

                              filled= True )

        

#Determinando la precisión del algoritmo

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

print(acc_decision_tree)
#predecir paciente enfermo





x_test = pd.DataFrame(columns=('target','sex', 'cp', 'fbs','restecg','exang','slope','ca','thal','kmean'))

x_test.loc[0] = (1,1,3,1,0,0,0,0,1,0)

y_pred = decision_tree.predict(x_test.drop(['target'], axis = 1))

print("Prediccion: " + str(y_pred))

y_proba = decision_tree.predict_proba(x_test.drop(['target'], axis = 1))

print("Probabilidad de acierto",y_proba[0][1]*100,'%')
# Predecir pacienta sano

x_test = pd.DataFrame(columns=('target','sex', 'cp', 'fbs','restecg','exang','slope','ca','thal','kmean'))

x_test.loc[0] = (0,1,2,1,1,0,1,1,1,1)

y_pred = decision_tree.predict(x_test.drop(['target'], axis = 1))

print("Prediccion: " + str(y_pred))

y_proba = decision_tree.predict_proba(x_test.drop(['target'], axis = 1))

print("Probabilidad de acierto",y_proba[0][0]*100,'%')
# Para probar la predicción pasamos el dataframe completo y cargamos el resultado en una variable

x_test = df6



y_pred = decision_tree.predict(x_test.drop(['target'], axis = 1))
# Añadimos la predicción(y_pred)como una columna al dataframe

df6['pred']=y_pred
df6.describe()
#Numeramos los pacientes usando el indice



a=[]

for i in range(0,303):

    a.append(i)



# Añadimos la columna de pacientes

df6['patient']=a





df6['patient'].describe()
#Representamos los datos reales

df6.plot(kind="scatter",x='patient' ,y="target",c='r')

# Representamos la predicción

df6.plot(kind="scatter",x='patient',y='pred')
