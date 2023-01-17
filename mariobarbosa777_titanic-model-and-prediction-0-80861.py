%reset -f 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib 

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn import preprocessing 
from sklearn import metrics
%matplotlib inline
#Leer Los datos
df_train = pd.read_csv("/kaggle/input/titanic/train.csv", index_col="PassengerId")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv", index_col="PassengerId")
#Borrar Variables
df_train.drop(columns=["Name","Cabin","Ticket"], inplace=True);
df_test.drop(columns=["Name","Cabin","Ticket"], inplace=True);
#Convertir a tipos de variable adecuados 
df_train.convert_dtypes();
df_test.convert_dtypes();
# Rellenar o Borrar valores NAN

df_train.fillna(df_train.median(), inplace=True)
#df_train.dropna(inplace=True)
df_test.fillna(df_train.median(), inplace=True)

#Reducir dimensionalidad encontrado correlación de variables
df_train["Family_members"] = df_train["SibSp"] + df_train["Parch"]
df_test["Family_members"] = df_test["SibSp"] + df_test["Parch"]
df_train.drop(columns=["SibSp","Parch"], inplace=True);
df_test.drop(columns=["SibSp","Parch"], inplace=True);
#Categorizar variables  (Sex y Embarked)
df_train[["Sex"]] = df_train[["Sex"]].astype("category")
df_test[["Sex"]] = df_test[["Sex"]].astype("category")

df_train[["Embarked"]] = df_train[["Embarked"]].astype("category")
df_test[["Embarked"]] = df_test[["Embarked"]].astype("category")

# Obtener dummis de variables categoricas 
df_train[["IsWomen","IsMan"]] = pd.get_dummies(df_train[["Sex"]])
df_test[["IsWomen","IsMan"]] = pd.get_dummies(df_test[["Sex"]])

df_train[["IsC","IsQ","IsS"]] = pd.get_dummies(df_train[["Embarked"]])
df_test[["IsC","IsQ","IsS"]] = pd.get_dummies(df_test[["Embarked"]])

# Borrar Dummies Sobrantes y Variables Categoricas Originales 
df_train.drop(columns=["IsMan","Sex","IsQ","Embarked"], inplace=True);
df_test.drop(columns=["IsMan","Sex","IsQ","Embarked"], inplace=True);
#Agrupar datos numericos 

#( ] ( ] ( ] ( ] ( ]

#0.8061
bins = [-10, 0, 5, 12, 32, 60, 100]
names = [1, 2, 3, 4, 5, 6] 

#Age 
df_train["Age"]=pd.cut( df_train["Age"], bins =bins, labels = names)
df_test["Age"]=pd.cut( df_test["Age"], bins =bins, labels = names )

df_train["Age"]=df_train.Age.astype('category').cat.codes
df_test["Age"]=df_test.Age.astype('category').cat.codes

df_train.Age.value_counts()

#df_train[df_train.Age==1]
#Graficas

#BoxPlot
sns.boxplot(data = df_train._get_numeric_data())
 
#pair map
sns.pairplot(df_train, hue="Survived", plot_kws={'alpha':0.3} )

#Graficas
# corr 
corr= df_train._get_numeric_data().corr()
sns.heatmap(corr, yticklabels = corr.columns, xticklabels = corr.columns,  annot=True)
# Machine Learning

# X = Features Y = Target 
X = df_train.drop(columns=['Survived'])
Y = df_train['Survived']

Classifiers_dict = {    "RandomForest"          :RandomForestClassifier(),
                        "GradientBoosting"      :GradientBoostingClassifier(),
                        "KNeighborsClassifier"  :KNeighborsClassifier(),
                        "DecisionTreeClassifier":DecisionTreeClassifier(),
                    } 
for  Name, Classifier  in Classifiers_dict.items():
    print(Name)
    print()
    CrossResults = cross_validate(Classifier, X, Y, cv=5, return_train_score=True)
    print (f"Train Mean = {np.mean(CrossResults['train_score'])}")
    print (f"Test Mean = {np.mean(CrossResults['test_score'])}" )
    print (f"Train MAX = {np.max(CrossResults['train_score'])}")
    print (f"Test MAX = {np.max(CrossResults['test_score'])}" )
    
    estimator = Classifier.fit(X,Y)
    Y_test=estimator.predict(df_test)


    try :
        print (list(zip(X, np.round(estimator.feature_importances_,decimals=4))))
    except:
        pass

    df_salida = df_test.copy(deep=True)
    df_salida['Survived'] = Y_test
    df_salida.to_csv(f"Prediction_Titanic_{Name}_.csv", index=True, columns=["Survived"])

    print("---"*100)

