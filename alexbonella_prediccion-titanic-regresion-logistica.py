import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv("../input/train.csv")
df_test=pd.read_csv("../input/test.csv")
train.info()
train.drop([889,890],inplace=True)
train.isnull().tail() # obtenemos una matrix boleana donde se muestra datos nulos 
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis') # eliminamos eje Y , leyenda de las barras
sns.set_style('whitegrid') # Estilo de la grilla 
sns.countplot(x='Survived',data=train,palette='viridis') 
sns.countplot(x='Survived',hue='Sex',data=train,palette='viridis') # si deseamos hacer un matizado con el genero 
# Ahora hagamos el mismo analisis pero con la clase social
sns.countplot(x='Survived',hue='Pclass',data=train)

train['Pclass'].value_counts() # Para evaluar el numero de pasajeros de la muestra 
sns.countplot(x='Pclass',data=train)
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
# Creamos la funcion de imputacion de datos para la edad

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
# Aplicamos nuevamente el heatmap para verificar si los datos fueron limpiados correctamente
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis') # de esta manera se observa los missing Data
# como podemos analizar hay muchos datos faltantes en las cabinas y quizas no sea un informacion importante
# entonces lo que hacemos es suprimir esta columna 
train.drop('Cabin',axis=1,inplace=True) # recordar que el implace es para hacer un guardado definitivo
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis') # de esta manera se observa los missing Data


train.drop(['Name','Embarked','SibSp','Ticket','Parch','Fare','Age'],axis=1,inplace=True)
train.head()


sex = pd.get_dummies(train['Sex'],drop_first=True) # con pandas caracterizamos las variables categoricas
#clase = pd.get_dummies(train['Pclass'],drop_first=True) # con pandas caracterizamos las variables categoricas
train = pd.concat([train,sex],axis=1)  # Realizamos concatenación para ingresar las variables convertidas
train.head()
train.drop(['Sex'],axis=1,inplace=True)
train.head()

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

X=train.drop('Survived',axis=1)
y=train['Survived'] 
X_test  = df_test.drop("PassengerId",axis=1).copy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.47, random_state=101)
from sklearn.linear_model import LogisticRegression 
lrmodel = LogisticRegression() # Creamos la instancia
lrmodel.fit(X_train,y_train)   # Ajustamos el modelo con los datos de entrenamiento
prediccion=lrmodel.predict(X_test) # Prediccion de los datos 
lrmodel.score(X_train, y_train)
logistic_score = round(lrmodel.score(X_train, y_train)*100,2)
logistic_score

from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier()
ranfor.fit(X_train, y_train)
pred_rf = ranfor.predict(X_test)

ranfor.score(X_train, y_train)

ranfor_score = round(ranfor.score(X_train, y_train)*100,2)
ranfor_score

df_final = pd.DataFrame({"Models": [ 'Logistic Regression',  'Random Forest'], 
                       "Score": [logistic_score, ranfor_score]})
df_final.sort_values(by= "Score", ascending=False)

from sklearn.metrics import classification_report # Nos indicara la precision de nuestro modelo
from sklearn.metrics import confusion_matrix
print (classification_report(y_test,prediccion))
confusion_matrix(y_test,prediccion)


print (classification_report(y_test,pred_rf))
confusion_matrix(y_test,pred_rf)



df_test.drop(['Name','Embarked','SibSp','Ticket','Parch','Fare','Age','Cabin','Sex','Pclass'],axis=1,inplace=True)
df_test.head()
final_report_LogR = pd.DataFrame({"PassengerId": df_test["PassengerId"], 
                       "Survived" : prediccion})

final_report_LogR.head()

final_report_LogR.to_csv('final_Pred_LogR.csv', index=False)
final_report_RanFor = pd.DataFrame({"PassengerId": df_test["PassengerId"], 
                       "Survived" : pred_rf})

final_report_RanFor.head()
final_report_RanFor.to_csv('final_Pred_RanFor', index=False)






