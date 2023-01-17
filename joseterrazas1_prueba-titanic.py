import pandas as pd

import numpy as np

import pandas_profiling



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import ensemble

from sklearn import svm

from sklearn import tree

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import mean_squared_error, r2_score   

from sklearn.externals import joblib



def nulos(data):

    print(data.shape)

    for col in data.columns:

        print(col,": ",data[col].isna().sum())

        

def outliers_iqr(ys):

    quartile_1, quartile_3 = np.percentile(ys, [25, 75])

    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * 1.5)

    upper_bound = quartile_3 + (iqr * 1.5)

    return np.where((ys > upper_bound) | (ys < lower_bound))



def limites_outliers_iqr(ys):

    quartile_1, quartile_3 = np.percentile(ys, [25, 75])

    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * 1.5)

    upper_bound = quartile_3 + (iqr * 1.5)

    limites = [lower_bound, upper_bound]

    return limites



def Ver_outliers(data, metodo):

    for col in data.columns:

        if (data.dtypes[col]=="int64") | (data.dtypes[col]=="float64"):

            a=metodo(data[col])

            print("Outlier de la columna ",col,": ",len(a[0]),"\n ", a[0])



def Ver_outliers_limites(data, metodo):

    for col in data.columns:

        if (data.dtypes[col]=="int64") | (data.dtypes[col]=="float64"):

            a=metodo(data[col])

            print("Outlier de la columna ",col,": ", a)#[0],"-",a[1])

            



def describe_by(data,cat):

    for c in data[cat].unique():

        dt=data[data[cat]==c]

        print("Descripcion de la categoria: ", c)

        print(dt.describe(include="all"))

        print()

        print()



def run_classifier(clf, X, y, num_tests=100):

    metrics = {'f1-score': [], 'precision': [], 'recall': [], 'accuracy': []}

    confusion = 0

    for _ in range(num_tests):

        

        ### INICIO COMPLETAR ACÁ 

        #### TIP: divida el dataset, entrene y genere las predicciones.

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, stratify=y)

    

        clf.fit(X_train,y_train)

        

        predictions=clf.predict(X_test)        

        ### FIN COMPLETAR ACÁ

        

        metrics['f1-score'].append(f1_score(y_test, predictions))  # X_test y y_test deben ser definidos previamente

        metrics['recall'].append(recall_score(y_test, predictions))

        metrics['precision'].append(precision_score(y_test, predictions))

        metrics['accuracy'].append(accuracy_score(y_test,predictions))

        confusion= confusion + np.array(confusion_matrix(y_test, predictions))

    return metrics, confusion/num_tests
import os

print(os.listdir("../input/titanic"))
#Cargar Data

train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")



print(train.shape)

train.head()
##Jugar con los nombres ("Mr.","Mrs.","Miss")

#Mr=np.where(("Mr." in train['Name']),1,0)

Mr=[]

Mrs=[]

Miss=[]

for n in train['Name']:

    if "Mr." in n:

        Mr.append(1)

        Mrs.append(0)

        Miss.append(0)

    elif "Mrs." in n:

        Mr.append(0)

        Mrs.append(1)

        Miss.append(0)

    elif "Miss" in n:

        Mr.append(0)

        Mrs.append(0)

        Miss.append(1)

    else:

        Mr.append(0)

        Mrs.append(0)

        Miss.append(0)



train['Mr']=Mr

train['Mrs']=Mrs

train['Miss']=Miss
pandas_profiling.ProfileReport(train)
features = ["Survived","Pclass","Sex","SibSp","Parch","Fare","Age"]#,"Mr","Mrs","Miss"]

train=train[features]



print(train.shape)

train.head()
##Imputacion edad por regresion



train_a=train[train['Age'].notnull()]

test_a=train[train['Age'].isnull()]



train_a.describe()#include='all')

#test_a.describe()
cols=["Pclass","Sex","SibSp","Parch","Fare"]#,"Mr","Mrs","Miss"]



x=pd.get_dummies(train_a[cols])

y=train_a.Age



regr=linear_model.LinearRegression()

regr.fit(x, y)

y_pred = regr.predict(x)



# Veamos los coeficienetes obtenidos

print('Coefficients: \n', regr.coef_)

# Este es el valor donde corta el eje Y (en X=0)

print('Independent term: \n', regr.intercept_)

# Error Cuadrado Medio

print("Mean squared error: %.2f" % mean_squared_error(y, y_pred))

# Puntaje de Varianza. El mejor puntaje es un 1.0

print('Variance score: %.2f' % r2_score(y, y_pred))



train_aux=pd.get_dummies(train_a)

test_aux=pd.get_dummies(test_a)

test_aux['Age'] = regr.predict(pd.get_dummies(test_a[cols]))



print(type(test_aux))

train = pd.concat([train_aux,test_aux])



print(train.shape)

train.describe()
#Eliminar outliers

#Ver_outliers(train,outliers_iqr)

Ver_outliers_limites(train, limites_outliers_iqr)



train.describe()
train = train[(train['SibSp']<3)&(train['Fare']<66)&(train['Age']>0)&(train['Age']<59)]

print(train.shape)

train.describe()
all_data = [train, test] # unimos bases

# Familiares viajando con la persona

for df in all_data:

    df['Familiares'] = df['SibSp'] + df['Parch']

# Si la persona viaja sola o no

for df in all_data:

    df['Viaja_Solo'] = np.where(df['Familiares']==0,1,0)

train.head()
dt=train.copy()

sns.set()

fig, ax = plt.subplots(figsize=(10,10))

ax=sns.heatmap(dt.corr(), square=True, annot=True)
x=train[['Pclass','SibSp','Parch','Fare','Viaja_Solo', 'Familiares','Sex_female','Sex_male','Age']]#,"Mr","Mrs","Miss"]]

y=train.Survived 


c0 = ("boost random forest:",ensemble.AdaBoostClassifier()) #otros clasificadores son  SVC, NuSVC and LinearSVC

c1 = ("Random forest:",ensemble.RandomForestClassifier()) #otros clasificadores son  SVC, NuSVC and LinearSVC

c2 = ("NB:", GaussianNB()) #otros clasificadores son  SVC, NuSVC and LinearSVC

c3 = ("DT:",DecisionTreeClassifier(criterion = 'gini', max_features= None))

c4 = ("LinearSVC:",svm.LinearSVC()) #otros clasificadores son  SVC, NuSVC and LinearSVC





classifiers = [c0, c1, c2, c3, c4]



for name, clf in classifiers:

    metrics, confusion = run_classifier(clf, x, y, num_tests=30)   # hay que implementarla en el bloque anterior.

    print("----------------")

    print("Resultados para clasificador: ",name) 

    print("accuracy: ", np.array(metrics['accuracy']).mean())

    print("Precision promedio:",np.array(metrics['precision']).mean())

    print("Recall promedio:",np.array(metrics['recall']).mean())

    print("F1-score promedio:",np.array(metrics['f1-score']).mean())

    print("matrix confusion promedio:\n", (confusion))

    print("----------------\n\n")
clasificador= ensemble.GradientBoostingClassifier()#ensemble.AdaBoostClassifier()

scores_clasificador = cross_val_score(clasificador,x,y,cv=10)

predict_clasificador = cross_val_predict(clasificador, x, y, cv=10)

print("Accuracy arbol cross-val 10-folds:", scores_clasificador.mean())

print(classification_report(y, predict_clasificador))