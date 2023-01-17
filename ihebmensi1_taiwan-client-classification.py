# 1- importation des bibliothéques pour la manipulation et la visualisation des données

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# 2- Importation de la base Bank of thaiwan



tai=pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv",index_col=0)

tai.head()
# Affichage de premier 5 client

tai.head()
#Afficher le nombre de chaque type de clients

tai["default.payment.next.month"].value_counts()
#Affichage de dimension du Dataframe

tai.shape
#la suppression des données manquants



tai5=tai.dropna()

data_propre=tai.dropna()

tai5.head(2)
#Regroupement des variables bill,paid et pay

bill = ['BILL_AMT1', 'BILL_AMT2',

       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

paid = ['PAY_AMT1',

       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

pay = ['PAY_0', 'PAY_2',

       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
#Afficher le nombre de chaque valeur dans Feature "EDUCATION"

tai5.EDUCATION.value_counts()
#Afficher le nombre de chaque valeur dans Feature "EDUCATION"

tai5.MARRIAGE.value_counts()
#Remplacer les variable non attendues dans les deux colonnes "EDUCATION" et "MARIAGE"    

fil = (tai5.EDUCATION == 5) | (tai5.EDUCATION == 6) | (tai5.EDUCATION == 0)

tai5.loc[fil, 'EDUCATION'] = 4



fil1 = (tai5.MARRIAGE== 0)

tai5.loc[fil1, 'MARRIAGE'] = 3
tai5.head(2)
#Affichage des colonnes de dataframe tai

tai.columns
#Faire centrer et réduire les valeurs des données tai5

col_to_norm = ['LIMIT_BAL','BILL_AMT1', 'BILL_AMT2',

       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',

       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

tai5[col_to_norm] = tai5[col_to_norm].apply(lambda x : (x-np.mean(x))/np.std(x))
tai5.head(2)
tai5.PAY_0.value_counts()
#Réaliser une analyse descriptive

data_propre.describe()
#Tester la variation entre la moyenne et la médiane

abs(   (data_propre.mean() - data_propre.median()) / (data_propre.mean()) )*100
# Visualiser les points abérants

data_propre.boxplot(figsize=(18,8));
# limiter 3 chiffres aprés la vigule

pd.options.display.float_format = '{:,.3f}'.format
# Affichage de la corrélation entre les variables

data_propre.corr()
# Visualiser la corrélation enutilisant heatmap sans les variables encodées

plt.figure(figsize=(25,15))

sns.heatmap(data_propre.corr() , annot=True)

plt.show
# Appliquer une fonction qui permet de la décomposition des données

# en train et test avec une division de 0.2

def get_data_splits(dataframe, valid_fraction=0.1):

    valid_fraction = 0.1

    valid_size = int(len(dataframe) * valid_fraction)



    train = dataframe[:-valid_size * 2]



    valid = dataframe[-valid_size * 2:-valid_size]

    test = dataframe[-valid_size:]

    

    return train, valid, test
from sklearn.feature_selection import SelectKBest, f_classif

# Suppression de variable cible "default payment next month"

feature_cols = tai5.columns.drop('default.payment.next.month')

train, valid, _ = get_data_splits(tai5)



# Appliquer la méthode "SelectKBest" en gardant que les 8 colonnes qui expliquent l'information le mieux



selector = SelectKBest(f_classif, k=8)



X_new = selector.fit_transform(train[feature_cols], train['default.payment.next.month'])

X_new
# Afficher dans un dataframe les meilleurs colonnes  , les autres colonnes ont des valeurs de 0

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train.index, 

                                 columns=feature_cols)

selected_features.head(2)
#Garder que les colonnes qui ont des valeurs non nuls, on obtient alors que les meilleurs colonnes 

selected_columns = selected_features.columns[selected_features.var() != 0]





train[selected_columns].head(2)
# Importer la classe "warnings" qui permet d'ignorer les erreurs de gravité "warnings"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

# On fait la division des données

train, valid, _ = get_data_splits(tai5)



X1, y1 = train[train.columns.drop("default.payment.next.month")], train['default.payment.next.month']



# On choisit lasso-régression pour filtrer les colonnes

logistic = LogisticRegression(C=1, penalty="l1", random_state=7).fit(X1, y1)

model = SelectFromModel(logistic, prefit=True)



X_new = model.transform(X1)

X_new
# Affichage des colonnes séléctionnés dans un dataframe

selected_features = pd.DataFrame(model.inverse_transform(X_new), 

                                 index=X1.index,

                                 columns=X1.columns)

#Suppression des donneés qui ont des valeurs nuls

selected_columns = selected_features.columns[selected_features.var()!=0]
selected_columns
# Découpage des données en variables explicatives et variables expliquées

X = tai5.iloc[:,:23]  

y = tai5.iloc[:,23]    
X.head(2)
y.values
# Importer la classe StandardScaler pour l'échantionnage

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
# Appliquer la méthode pour le dataframe x des variables explicatives

ss.fit(X.values)

matriceTCL = ss.transform(X.values)

#Afficher les données aprés remise a l'echellle avec standarscaler

dataTCL =pd.DataFrame(matriceTCL , columns=X.columns)

dataTCL.head(2)
#Afficher la forme des données de chaque feature en utilisant "hist"

pd.DataFrame(matriceTCL).hist(figsize=(15,13));
# Importer la classe ExtraTreesClassifier

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt
#Diviser les données pour l'aprentissage et le test afin d'avoir une meilleur résultat

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#Faire l'apprentissage des données en utilisant le model ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(X_train,y_train)

#Afficher "feature_importances" pour chaque colonne

print(model.feature_importances_) 
#Visualiser "feature_importance" dans un graphe

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)

feat_importances.nlargest(10).plot(kind='barh',color=['pink','black', 'coral', 'blue','red', 'green', 'cyan','y','orange','lime'])

plt.show()
# Importer la classe de PCA

from sklearn.decomposition import PCA

#initialiser la méthode PCA

pca = PCA(0.95)



# fit and transform les données explicatives seulement

datapca = pca.fit(dataTCL.iloc[:,1:])

# Afficher les pourcentages de variance expliquée de chaque feature

val = pd.Series(datapca.explained_variance_ratio_)

val.plot(kind='bar', title="graphes des valeurs propres")

plt.show()

#  Création d'un dataframe qui contient les valeurs de chaque composants 

# Initiation pour afficher la cercle de corrélation

coef = np.transpose(pca.components_)

cols = ['PC-'+str(x) for x in range(len(val))]

pc_infos = pd.DataFrame(coef, columns=cols, index=dataTCL.iloc[:,1:].columns)

pca.n_components_

datapca.explained_variance_ratio_
# Affichage de la cercle de corrélation

# Réalisation d'un graphe qui contient un cercle

plt.Circle((0,0),radius=10, color='g', fill=False)

circle1=plt.Circle((0,0),radius=1, color='g', fill=False)

# Ajouter les axes et donner la limite pour chaque axe

fig, axes= plt.subplots(figsize=(10,10))

axes.set_xlim(-1,1)

axes.set_ylim(-1,1)

fig.gca().add_artist(circle1)

plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)

plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

axes.add_artist(circle1)

# Affichage de chaque variable explicative dans la cercle

for idx in range(len(pc_infos["PC-0"])):

    x = pc_infos["PC-0"][idx]

    y = pc_infos["PC-1"][idx]

    plt.plot([0.0,x],[0.0,y],'k-')

    plt.plot(x, y, 'rx')

    plt.annotate(pc_infos.index[idx], xy=(x,y))

plt.xlim((-1,1))

plt.ylim((-1,1))

plt.title("Circle of Correlations")
tai5.columns
clos=['LIMIT_BAL', 'PAY_0', 'PAY_2',

       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

tai5[clos].head(2)
y = tai5.iloc[:,23] 
y.values
#Diviser les données pour l'apprentissage et le test

from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(tai5[clos].values,y.values , test_size = 0.2,random_state=0)
from sklearn.neighbors import KNeighborsClassifier

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

error = []



for i in range(1, 40):

    knn = KNeighborsClassifier(i)

    knn_model = knn.fit(X_train1, y_train1)

    pred_i = knn_model.predict(X_test1)

    error.append(np.mean(pred_i != y_test1))

plt.figure(figsize=(12, 6))

plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',

         markerfacecolor='blue', markersize=10)

plt.title('Taux Erreur pour les differentes valeurs de k')

plt.xlabel('K ')

plt.ylabel('Erreur')
from sklearn.neighbors import KNeighborsClassifier

# Appliquer la meilleur valeur de k=2

knn1 = KNeighborsClassifier(28)

# Faire l'étape d'apprentissage  

knn_model1 = knn.fit(X_train1, y_train1)

# réaliser la prédiction de X_test1

y_pred_knn1 =knn_model1.predict(X_test1)

# Afficher l'accuracy de prédiction

knn_score=knn_model1.score(X_test1,y_test1)

knn_score
# Importer la classe accuracy_score

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test1, y_pred_knn1))
#Appliquer la matrice de confusion

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test1, y_pred_knn1))
#Afficher "classification_report"

from sklearn.metrics import classification_report

print(classification_report(y_test1, y_pred_knn1))
#Importer la classe "DecisionTreeClassifier"

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

# Faire l'étape d'apprentissage  

dtc_model = DecisionTreeClassifier().fit(X_train1, y_train1)

# réaliser la prédiction de X_test1

y_pred_dtc = dtc_model.predict(X_test1)

# Afficher l'accuracy de prédiction

dtc_score=dtc_model.score(X_test1,y_test1)

dtc_score
# Importer la classe accuracy_score

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test1, y_pred_dtc))
#Afficher "classification_report"

from sklearn.metrics import classification_report

print(classification_report(y_test1, y_pred_dtc))
#Importer la classe "RandomForestClassifier"

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=30,

 min_samples_leaf= 1,

 min_samples_split= 2,

 n_estimators= 100)

# Faire l'étape d'apprentissage  

rfc_model = rfc.fit(X_train1, y_train1)

# réaliser la prédiction de X_test1

y_pred_rfc = rfc_model.predict(X_test1)

# Afficher l'accuracy de prédiction

rfc_score=rfc_model.score(X_test1,y_test1)

rfc_score
# Importer la classe accuracy_score

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test1, y_pred_rfc))
#Appliquer la matrice de confusion

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test1, y_pred_rfc))
#Afficher "classification_report"

from sklearn.metrics import classification_report

print(classification_report(y_test1, y_pred_rfc))
#Importer la classe "LogisticRegression"

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

# Faire l'étape d'apprentissage  

lr_model = lr.fit(X_train1, y_train1)

# réaliser la prédiction de X_test1

y_pred_lr = lr_model.predict(X_test1)

# Afficher l'accuracy de prédiction

lr_score=lr_model.score(X_test1,y_test1)

lr_score
#Appliquer la matrice de confusion

print(confusion_matrix(y_test1, y_pred_lr))
#Afficher "classification_report"

print(classification_report(y_test1, y_pred_lr))
#Tester la prédiction d'un nouveau donnée

df = np.array([20000,2,2,1,24,2,2,-1,-1,-1,-1,3919,3102,689,0,0,0,0,689,0,0,0,0]).reshape(1,23)

df2 = np.array([20000,2,2,2,4,6,1]).reshape(1,7)

lr_model.predict(df2)
y_pred_lr
#Importer la classe "GaussianNB"

from sklearn.naive_bayes import GaussianNB

model_NB= GaussianNB()

# Faire l'étape d'apprentissage  

model_naive=model_NB.fit(X_train1,y_train1)

# réaliser la prédiction de X_test1

y_pred_nb =model_naive.predict(X_test1)

# Afficher l'accuracy de prédiction

nb_score=model_naive.score(X_test1,y_test1)

nb_score
#Appliquer la matrice de confusion

print(confusion_matrix(y_test1, y_pred_nb))
#Afficher "classification_report"

print(classification_report(y_test1, y_pred_nb))
#Importer la classe "SVC"

from sklearn.svm import SVC

model_svm= SVC(gamma='auto',C= 20, kernel='rbf')

# Faire l'étape d'apprentissage 

model_svm1=model_svm.fit(X_train1,y_train1)

# réaliser la prédiction de X_test1

y_pred_svm = model_svm.predict(X_test1)

# Afficher l'accuracy de prédiction

svm_score=model_svm1.score(X_test1,y_test1)

svm_score
#Appliquer la matrice de confusion

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test1, y_pred_svm))
#Afficher "classification_report"

print(classification_report(y_test1, y_pred_svm))
#Tester la prédiction d'un nouveau donnée

df = np.array([210000,1,1,2,29,-2,-2,-2,-2,-2,-2,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(1,23)

df2 = np.array([20000,2,2,2,4,6,1]).reshape(1,7)





model_svm1.predict(df2)
#Importer la classe "XGBClassifier"

from xgboost import XGBClassifier

model_xgboost= XGBClassifier()

# Faire l'étape d'apprentissage 

model_xgboost.fit(X_train1,y_train1)

# réaliser la prédiction de X_test1

y_pred_xgb = model_xgboost.predict(X_test1)

# Afficher l'accuracy de prédiction

model_xgboost.score(X_test1,y_test1)
xgb_score=model_xgboost.score(X_test1,y_test1)

xgb_score
#Appliquer la matrice de confusion

print(confusion_matrix(y_test1, y_pred_xgb))
#Afficher "classification_report"

print(classification_report(y_test1, y_pred_xgb))
#Tester la prédiction d'un nouveau donnée

df = np.array([20000,2,2,1,24,2,2,-1,-1,-1,-1,3919,3102,689,0,0,0,0,689,0,0,0,0]).reshape(1,23)

df2 = np.array([20000,2,2,2,4,6,1]).reshape(1,7)

model_xgboost.predict(df2)
#Importer la classe "AdaBoostClassifier"

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

model_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),

                         algorithm="SAMME",

                         n_estimators=500, learning_rate=0.8)

# Faire l'étape d'apprentissage 

model_ada.fit(X_train1, y_train1)

# réaliser la prédiction de X_test1

y_pred_ada = model_ada.predict(X_test1)

model_ada.score(X_test1,y_test1)

# Afficher l'accuracy de prédiction

ada_score=model_ada.score(X_test1,y_test1)

ada_score
#Appliquer la matrice de confusion

print(confusion_matrix(y_test1, y_pred_ada))
#Afficher "classification_report"

print(classification_report(y_test1, y_pred_ada))
#Tester la prédiction d'un nouveau donnée

df = np.array([20000,2,2,1,24,2,2,-1,-1,-1,-1,3919,3102,689,0,0,0,0,689,0,0,0,0]).reshape(1,23)

df2 = np.array([20000,2,2,2,4,6,1]).reshape(1,7)

model_ada.predict(df)
### Courbe ROC de différents modéle
#Importer les méthodes "roc_curve, auc"

%matplotlib inline



from sklearn.metrics import roc_curve, auc
# Création d'un tuple qui contient false positive rate, trus positive rate 

fpr1, tpr1, threshold1 = roc_curve(y_test1, y_pred_dtc) 

roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, threshold2 = roc_curve(y_test1, y_pred_rfc) 

roc_auc2 = auc(fpr2, tpr2)

fpr3, tpr3, threshold3 = roc_curve(y_test1, y_pred_knn1)

roc_auc3 = auc(fpr3, tpr3)

fpr4, tpr4, threshold4 = roc_curve(y_test1, y_pred_svm)

roc_auc4 = auc(fpr4, tpr4)

fpr5, tpr5, threshold5 = roc_curve(y_test1, y_pred_xgb)

roc_auc5 = auc(fpr5, tpr5)

fpr6, tpr6, threshold6 = roc_curve(y_test1, y_pred_ada)

roc_auc6= auc(fpr6, tpr6)

fpr7, tpr7, threshold7 = roc_curve(y_test1, y_pred_lr)

roc_auc7= auc(fpr7, tpr7)

fpr8, tpr8, threshold8 = roc_curve(y_test1, y_pred_nb)

roc_auc8= auc(fpr8, tpr8)
plt.figure(figsize=(10,10)) 

plt.plot(fpr1, tpr1, color='navy', lw=2, label='CART ROC curve (area = %0.2f)'% roc_auc1)

plt.plot(fpr2, tpr2, color='green', lw=2, label='Random Forest ROC curve (area = %0.2f)'% roc_auc2)

plt.plot(fpr3, tpr3, color='yellow', lw=2, label='kNN ROC curve (area = %0.2f)'% roc_auc3)

plt.plot(fpr4, tpr4, color='orange', lw=2, label='SVM ROC curve (area = %0.2f)'% roc_auc4)

plt.plot(fpr5, tpr5, color='purple', lw=2, label='XGBOOST ROC curve (area = %0.2f)'% roc_auc5)

plt.plot(fpr6, tpr6, color='black', lw=2, label='ADABOOST ROC curve (area = %0.2f)'% roc_auc6)

plt.plot(fpr7, tpr7, color='lime', lw=2, label='LogisticR ROC curve (area = %0.2f)'% roc_auc7)

plt.plot(fpr8, tpr8, color='cyan', lw=2, label='NaiveB ROC curve (area = %0.2f)'% roc_auc8)

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--') 

plt.xlim([0.0, 1.0]) 

plt.ylim([0.0, 1.05]) 

plt.xlabel('False Positive Rate') 

plt.ylabel('True Positive Rate') 

plt.title('Classifiers ROC curves') 

plt.legend(loc = "lower right")

plt.show()
# Création de fichier pickle  contient les models traités au-dessus

import pickle

with open('taiwan_final.pkl','wb') as file:

    pickle.dump([knn_model1,dtc_model,rfc_model,lr_model,model_naive,model_svm1,model_xgboost,model_ada],file)
# Création de fichier pickle contient false positive rate, true positive rate et roc_auc

import pickle

with open('taiwan_final_roc.pkl','wb') as file:

    pickle.dump([fpr1, tpr1,roc_auc1,fpr2, tpr2,roc_auc2,fpr3, tpr3,roc_auc3,fpr4, tpr4,roc_auc4,fpr5, tpr5,roc_auc5,fpr6, tpr6,roc_auc6,fpr7, tpr7,roc_auc7,fpr8, tpr8,roc_auc8],file)
# Création de fichier pickle contient le score de chaque modéle traité

import pickle

with open('taiwan_final_score.pkl','wb') as file:

    pickle.dump([knn_score,dtc_score,rfc_score,lr_score,nb_score,svm_score,xgb_score,ada_score],file)
#ouvrir un fichier pickle

with open('taiwan_final.pkl','rb') as f:

    ma= pickle.load(f)

ma[0]
# Importer la classe kmeans

from sklearn.cluster import KMeans

#Utilise la méthode elbow pour connaitre la meilleur valeur de k

sse = []

k_rng = range(1,10)

for k in k_rng:

    km = KMeans(n_clusters=k)

    km.fit(X)

    sse.append(km.inertia_)

    print (km.inertia_)

plt.xlabel('K')

plt.ylabel('Sum of squared error')

plt.plot(k_rng,sse)
#Appliquer la méthode Kmeans pour k=2

Model1 = KMeans(n_clusters=2)

Model1.fit(data_propre.iloc[:,1:])

Model1.labels_
#faire identifier la classe de chaque valeur en utilisant crosstab

pd.crosstab(y,Model1.labels_)
## CAH
#Importer la classe CAH

from scipy.cluster.hierarchy import dendrogram, linkage,fcluster,set_link_color_palette

# Faire la liaison entre tous les données en utilisant la fonction linkage()

Z= linkage(X,method='ward',metric='euclidean')
plt.title("CHA") 

#Afficher la liaison obtenu avec dendogramme

dendrogram(Z,labels=X.index,orientation='left',color_threshold=0) 

plt.show()
# Choisir un threshhold qui permet de diviser les données en deux classes

plt.title('CAH avec matérialisation des 2 classes') 

dendrogram(Z,labels=X.index,orientation='left',color_threshold=1500

          ) 

plt.show()
#Appliquer la ségmentation en utilisant fcluster

clusters = fcluster(Z,criterion='distance', t=1500)
#faire identifier la classe de chaque valeur en utilisant crosstab

pd.crosstab(y,clusters)
# Importer la classe DBSCAN,metric

from sklearn.cluster import DBSCAN 

from sklearn import metrics 

from sklearn.datasets.samples_generator import make_blobs 



  

# Appliquer DBSCAN 

db = DBSCAN(eps=0.3, min_samples=10).fit(X) 

core_samples_mask = np.zeros_like(db.labels_, dtype=bool) 

core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_ 

  

# Identifier le nombre de clusters   

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 

  

print(labels)
X.values
#Appliquer DBSCAN au données échantilooné

DBSCANModel = DBSCAN(metric='euclidean',eps=0.25,min_samples=10,algorithm='auto').fit(X.values)

DBSCANModel