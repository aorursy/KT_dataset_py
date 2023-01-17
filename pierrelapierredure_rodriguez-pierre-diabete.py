import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


%matplotlib inline

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


diab = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
diab.head().T
#CONDITIONNEMENT : Pas de colonnes inutiles mais des données absentes avec 0, qu'il faut remplacer :
def replace_0(df,col) :
    df1 = df.copy()
    n = df.shape[0]
    m = df[col].mean()
    s = df[col].std()
    for i in range(n) :
        if df.loc[i,col]==0 :
            df1.loc[i,col] = np.random.normal(m,s)
    return df1

d1 = replace_0(diab,'Glucose')
d2 = replace_0(d1,'BloodPressure')
d3 = replace_0(d2,'SkinThickness')
d4 = replace_0(d3,'Insulin')
d5 = replace_0(d4,'BMI')
d5
#Recherche de données aberrantes, normalisation si besoin..
plt.hist(d5.Glucose, bins=80)
plt.hist(d5.BloodPressure, bins=80)
#etc ...
#Partie intéressante : apprentissage pour prédiction
X = d5.drop(['Outcome'], axis=1)
y = d5.Outcome

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#La régression :
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)

y_lr = lr.predict(X_test)

#Test de succès 
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

print(confusion_matrix(y_test,y_lr)) #VP, VN, FP, FN
print(accuracy_score(y_test,y_lr)) #Sous forme de %
print(classification_report(y_test, y_lr)) #Affichage plus lisible


#Traitement de ces probas plus ergonomique : (on peut garder les même commandes que pour le titanic)
probas = lr.predict_proba(X_test)
dfprobas = pd.DataFrame(probas,columns=['proba_0','proba_1'])
dfprobas['y'] = np.array(y_test)
dfprobas

plt.figure(figsize=(10,10))
sns.distplot(1-dfprobas.proba_0[dfprobas.y==0], bins=50)
sns.distplot(dfprobas.proba_1[dfprobas.y==1], bins=50)

#On construit la courbe ROC
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1]) 
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)
plt.figure(figsize=(12,12))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')        # plus mauvaise courbe
plt.plot([0,0,1],[0,1,1],'g:')     # meilleure courbe
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#On peut aussi tester avec une forêt :
from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
print(classification_report(y_test, y_rf))
cm = confusion_matrix(y_test, y_rf)
print(cm)
rf1 = ensemble.RandomForestClassifier(n_estimators=10, min_samples_leaf=10, max_features=3)
rf1.fit(X_train, y_train)
y_rf1 = rf.predict(X_test)
print(classification_report(y_test, y_rf1))

#On peut tracer la courbe :
from sklearn.model_selection import validation_curve
params = np.arange(1, 300,step=30)
train_score, val_score = validation_curve(rf, X, y, 'n_estimators', params, cv=7)
plt.figure(figsize=(12,12))
plt.plot(params, np.median(train_score, 1), color='blue', label='training score')
plt.plot(params, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('n_estimators')
plt.ylabel('score');
#On peut optimiser le choix des paramètres :
from sklearn import model_selection
param_grid = {
              'n_estimators': [10, 100, 500],
              'min_samples_leaf': [1, 20, 50]
             }
estimator = ensemble.RandomForestClassifier()
rf_gs = model_selection.GridSearchCV(estimator, param_grid)

rf_gs.fit(X_train, y_train) #Entraînement
rf2 = rf_gs.best_estimator_ #Meilleur estimateur ?
y_rf2 = rf2.predict(X_test)
# On a accès à l'importance des paramètres :
importances = rf2.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8,5)) # VISUALISATION
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), X_train.columns[indices])
plt.title('Importance des caracteristiques') 

# Option : utiliser XGboost, très efficace 
