import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid', {"axes.grid" : False})

sns.set_context('notebook')

np.random.seed(42)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.externals import joblib

from sklearn import datasets, linear_model, metrics

from mlxtend.plotting import plot_decision_regions

baseDados = pd.read_csv('diabetes.csv')

baseDados.head()
baseDados_copy = baseDados.copy(deep = True)

baseDados_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = baseDados_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(baseDados_copy.isnull().sum())
baseDados_copy['Glucose'].fillna(baseDados_copy['Glucose'].mean(), inplace = True)

baseDados_copy['BloodPressure'].fillna(baseDados_copy['BloodPressure'].mean(), inplace = True)

baseDados_copy['SkinThickness'].fillna(baseDados_copy['SkinThickness'].median(), inplace = True)

baseDados_copy['Insulin'].fillna(baseDados_copy['Insulin'].median(), inplace = True)

baseDados_copy['BMI'].fillna(baseDados_copy['BMI'].median(), inplace = True)
p=sns.pairplot(baseDados_copy, hue = 'Outcome')

plt.figure(figsize=(12,10)) 

p=sns.heatmap(baseDados.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X =  pd.DataFrame(sc_X.fit_transform(baseDados_copy.drop(["Outcome"],axis = 1),),

        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = baseDados_copy.Outcome
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier





test_scores = []

train_scores = []



for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(X_train,y_train)

    

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test)) 
max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
#resultado em grafico

plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')
knn = KNeighborsClassifier(11) #Incluindo a classificação 11 porque o código de max_test_score indicou o K=11



knn.fit(X_train,y_train)

knn.score(X_test,y_test)
#from sklearn.model_selection import GridSearchCV



#Knn = KNeighborsClassifier()



#param_grid = {'n_neighbors': np.arange(1, 100)}



#Knn_gscv = GridSearchCV(Knn, param_grid, cv=5)#fit model to data

#Knn_gscv.fit(X_train,y_train)



#Knn_gscv.best_params_
#import confusion_matrix

from sklearn.metrics import confusion_matrix

#let us get the predictions using the classifier we had fit above

y_pred = knn.predict(X_test)

confusion_matrix(y_test,y_pred)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
#Falsos e verdadeiros positivos - mapa

y_pred = knn.predict(X_test)

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
#import classification_report

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
#CURVA DE PROBABILIDADE

from sklearn.metrics import roc_curve

y_pred_proba = knn.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)



plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Knn(n_neighbors=11) ROC curve')

plt.show()
regr = linear_model.LinearRegression()



baseDados_copy = LogisticRegression() #criando instancia chamada diabetesCheck

baseDados_copy.fit(X, y)

accuracy = baseDados_copy.score(X, y) #encontrar a acurácia

print("accuracy = ", accuracy * 100, "%")

#import GridSearchCV

from sklearn.model_selection import GridSearchCV

#In case of classifier like knn the parameter to be tuned is n_neighbors

param_grid = {'n_neighbors':np.arange(1,100)}

knn = KNeighborsClassifier()

knn_cv= GridSearchCV(knn,param_grid,cv=5)

knn_cv.fit(X,y)



print("Best Score:" + str(knn_cv.best_score_))

print("Best Parameters: " + str(knn_cv.best_params_))
from sklearn.model_selection import cross_val_score

Knn_cv = KNeighborsClassifier(n_neighbors=25)





cv_scores = cross_val_score(Knn_cv, X, y, cv=5)#print each cv score (accuracy) and average them

print(cv_scores)

print('cv_scores mean:{}'.format(np.mean(cv_scores)))
