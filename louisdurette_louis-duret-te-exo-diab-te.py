# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline
# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
# SeaBorn : librairie de graphiques avancés
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
t = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
# Affichage des 10 premiers éléments du tableau
t.head(10)
t = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
# créer une valeur aléatoire suivant la loi normale et les valeurs de la moyenne et de l'écart-type
def replace_0(df,col) :
    df1 = df.copy()
    df1[col] = df[col].replace(0,np.random.normal(df[col].mean(),df[col].std()))
    return df1
t = replace_0(t,'Glucose')
t = replace_0(t,'BloodPressure')
t = replace_0(t,'SkinThickness')
t = replace_0(t,'Insulin')
t = replace_0(t,'BMI')
t[t.Glucose==0]
t[t.BloodPressure==0]
t[t.SkinThickness==0]
t[t.Insulin==0]
t[t.BMI==0]
#Caractéristiques
X = t.drop(['Outcome'], axis=1)
#Résultats
y = t.Outcome
#Importation de méthode pour la séparation des ensembles
from sklearn.model_selection import train_test_split
#Séparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#Vérification du bon fonctionnement séparation entre l'apprentissage et le test
print(X_train.shape)
print(X_test.shape)
#Importation de la méthode de régression logistique
from sklearn.linear_model import LogisticRegression
#Entrainement
lr = LogisticRegression()
lr.fit(X_train,y_train)
#Prédiction
y_lr = lr.predict(X_test)
# Importation des méthodes de mesure de performances
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
#Calcul du score
rf_score = accuracy_score(y_test, y_lr)
print(rf_score)
#Matrice de confusion
print(confusion_matrix(y_test,y_lr))
pd.crosstab(y_test, y_lr, rownames=['Reel'], colnames=['Prediction'], margins=True)
print(classification_report(y_test, y_lr))
t.Outcome.value_counts()
#Importation de la méthode randon forests
from sklearn import ensemble
#Entrainement
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
#Prédiction
y_rf = rf.predict(X_test)
#Calcul du score
rf_score = accuracy_score(y_test, y_rf)
print(rf_score)
#Matrice de confusion
cm = confusion_matrix(y_test, y_rf)
print(cm)
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
print(classification_report(y_test, y_rf))
from sklearn import model_selection
param_grid = {
              'n_estimators': [10, 100, 500],
              'min_samples_leaf': [1, 20, 50],
              'max_features': [1, 2, 4, 8]
             }
estimator = ensemble.RandomForestClassifier()
rf_gs = model_selection.GridSearchCV(estimator, param_grid)
# Détermination du meilleur groupe de paramètres
rf_gs.fit(X_train, y_train)
#Affichage du meilleur groupe de paramètre
print(rf_gs.best_params_)
rf2 = rf_gs.best_estimator_
y_rf2 = rf2.predict(X_test)
rf_score = accuracy_score(y_test, y_rf2)
print(rf_score)
print(classification_report(y_test, y_rf2))
# Sous Jupyter, si xgboost n'est pas déjà installé
!pip install xgboost
#Importation de la méthode
import xgboost as XGB
#Entrainement
xgb  = XGB.XGBClassifier()
xgb.fit(X_train, y_train)
#Prédiction
y_xgb = xgb.predict(X_test)
#Calcul du score
rf_score = accuracy_score(y_test, y_xgb)
print(rf_score)
#Matrice de confusion 
cm = confusion_matrix(y_test, y_xgb)
print(cm)
#Classification report
print(classification_report(y_test, y_xgb))
importances = rf2.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), X_train.columns[indices])
plt.title('Importance des caracteristiques')