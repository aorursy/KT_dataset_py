# Import des libraries classique (numpy, pandas, ...)
import pandas as pd
import numpy as np
import re
import sklearn as sk
from matplotlib import pyplot as plt
plt.style.use('ggplot')
!pip install seaborn --upgrade
import seaborn as sb

print(sb.__version__)
# Import du dataframe "data.csv"
df = pd.read_csv("../input/prediction-of-data-scientist-profiles/data.csv")
df_or = df.copy()
df.head(10)
print("Total number of observations: {}  \n".format(df.shape[0]))
print("The sum of missing values by variables:  \n")
print(df.isna().sum())
# remplacer les ',' par des '.' pour bien convertir les chiffres après la virgule dans la variable "Experience"
df['Experience'] = df.Experience.str.replace(',', '.').astype(float)

# remplire les valeurs manquantes dans la variable "Experience" par la valeur médiane pour les data scientists
df_DS =  df.loc[df['Metier']=='Data scientist','Experience']
df_DS = df_DS.fillna(df_DS.median())
df.loc[df['Metier']=='Data scientist','Experience'] = df_DS.copy() 
print('Missing values in Experience for data scientists', df.loc[df['Metier']=='Data scientist','Experience'].isnull().sum())

# remplire les valeurs manquantes dans la variable "Experience" par la valeur médiane pour les data engineers
df_DS =  df.loc[df['Metier']=='Data scientist','Experience']
df_DS = df_DS.fillna(df_DS.median())
df.loc[df['Metier']=='Data scientist','Experience'] = df_DS.copy() 
print('Missing values in Experience for data engineers', df.loc[df['Metier']=='Data scientist','Experience'].isnull().sum())

df_mean =  df[df['Metier'].isin(['Data scientist','Data engineer','Lead data scientist'])]
df_mean.groupby('Metier')['Experience'].mean()

df_Ex = df.groupby('Metier')['Experience'].mean()
df_mean = [df_Ex.get(k) for k in  df['Metier'].value_counts().keys()] 
#df_Ex = df.groupby('Metier')['Experience'].mean()
x=-0.6; y=0.25
ax = pd.Series(df_mean, index= df['Metier'].value_counts().keys()).plot(kind='barh')
totals =[] 
for p in ax.patches:
    totals.append(p.get_width())
    total = np.sum(totals)    
for ix,i in enumerate(ax.patches, 0):
    ax.text(i.get_width()+x, i.get_y()+y, str(round(df_mean[ix],3)), weight='bold', color='black')
plt.xlabel("Année d'experiences en moyenne")
plt.show()
# Plot histogramme de variable Expérience pour déterminer sa distribution et choisi la bonne méthode de découpage
plt.figure(figsize=(10,8))
sb.histplot(df,x='Experience', kde=True, ) 
plt.show()
labels = ['débutant', 'confirmé', 'avancé', 'expert']
Exp_level = pd.cut(df['Experience'], bins=4, labels= labels)
df["Exp_level"] = pd.Series(Exp_level, index=df.index)
# Affichage des résultats
ax = pd.Series(df['Exp_level'].value_counts().values, index=df['Exp_level'].value_counts().keys()).plot.barh(figsize=(10,7))
plt.xlabel("# de profile")
plt.subplots_adjust(wspace=0.3, hspace=0.2)
ax.invert_yaxis()
#Split la variable technologies pour voir chaque technologie
df_Tq = pd.DataFrame(df.Technologies.str.split('/').tolist()).stack()
df_Tq  = pd.DataFrame(df_Tq)
df_Tq.columns = ['Technologies']

plt.figure(figsize=(15,12))
df_Tq['Technologies'].value_counts().plot(kind='bar')
plt.show()

print('Les 5 technologies les plus utilisées sont:')
df_Tq['Technologies'].value_counts()[0:5]
    
#step1: Transformation les variables catégorielles

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_enc = df.loc[:,['Experience']]
#df_enc['Ville_enc'] = pd.Series(le.fit_transform(np.squeeze(df.loc[:,['Ville']])[:]), index = df_enc.index)
df_enc['Technologies_enc'] = pd.Series(le.fit_transform(np.squeeze(df.loc[:,['Technologies']])[:]), index = df_enc.index)
df_enc['Diplome_enc'] = pd.Series(le.fit_transform(np.squeeze(df.loc[:,['Diplome']])[:]), index = df_enc.index)
df_enc = np.round(df_enc,2)

print(df_enc.head(10))
print(df_enc.info())
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

#remplire les valeurs qui sont encore encore manquantes dans la variable "Experience" 
mean_da = df_enc['Experience'].dropna().mean()
df_enc['Experience'] = df_enc['Experience'].fillna(mean_da)


X = df_enc.astype(float)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, n_init=100 , max_iter=5000,tol=1e-8,random_state=0).fit(X_scaled)
labels_ = kmeans.labels_
#Ajout des clusters identifiés aux données d'origine
df_enc['Cluster'] = pd.Series(labels_)
df_plot = df.copy()
df_plot['Cluster'] = pd.Series(labels_)
# Evaluation de l'erreur de clustering par: Indicateur de compacité des classes(la dispersion à l’intérieur de chaque groupe).
c1, c2 = kmeans.cluster_centers_
Dist = lambda i,j: 100 * ((i-j)**2).sum() / ((i)**2).sum()
quad_dist1 = Dist(X_scaled[labels_==0], c1) 
quad_dist2 = Dist(X_scaled[labels_==1], c2) 
print('La métrique utilisée est indicateur de compacité des classes pour chaque cluster: {0:2.2f}% et {1:2.2f} %'.format(quad_dist1, quad_dist2))
print("les erreurs de clustering obtenus sont < 20% ")
# Pour l’évaluation intrinsèque, je choisis le coefficient de silhouette :
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_))
print("On remarque Coefficient Silhouette ~1 et <<-1")

print("On remarque aussi que deux technologies revient pus souvent dans le cluster 1 (resp. cluster 2) qui sont : ")
print(df.loc[labels_==0,'Technologies'].value_counts()[0:2])
print(df.loc[labels_==1,'Technologies'].value_counts()[0:2])

print("Les caracteristiques imortantes des deux clusters sont le 'Diplome':\
  \n  'Master' & 'Bachelor' pour le deuxieme cluster et 'No diplome' & 'PhD' pour le  premier cluster.")
# Affichage des résultats
print("\n Affichage des caractéristiques du cahque cluster : \n")      
a = ['Experience', 'Technologies',  'Diplome']# ,'Ville']
sb.set_theme(style="ticks", color_codes=True)

for n,i in enumerate(a,1):
    plt.figure(figsize=(20,30))
    plt.subplot(521),
    ax = df_plot.loc[labels_==0,i].value_counts().plot(kind='bar', color = '#5c3a97ff', label='cluster_2')
    plt.legend()
    plt.xlabel(i)
    plt.subplot(522),
    ax = df_plot.loc[labels_==1,i].value_counts().plot(kind='bar', color = '#fdb861ff', label='cluster_1')
    plt.ylabel("# de profile")
    plt.xlabel(i)
    plt.legend()
plt.show()
    
from sklearn.linear_model import LogisticRegressionCV
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
le = LabelEncoder()
mask = df.loc[:,'Metier'].isna() 
X = np.array(df_enc[~mask])
y_m = df.loc[~mask,'Metier']
y = le.fit_transform(df.loc[~mask,'Metier'].dropna())
Metier_test = np.array(df_enc[mask])
y_lab=[]
for i in [0,1,2,3]:
    y_lab.append(y_m[y==i].unique()[0])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.25, shuffle=True, random_state=40)
x_ax = range(len(y_test))

#LogisticRegressionCV
clf = LogisticRegressionCV(cv=5, random_state=0, max_iter=400, multi_class='multinomial').fit(X_train, y_train)

y_predicted = clf.predict(X_test)
print('Precision de classification par LogisticRegression : '+str(np.round(accuracy_score(y_test, y_predicted),2)*100)+'%   \n')
print(classification_report(y_test, y_predicted)) 
cm = confusion_matrix(y_test, y_predicted)
print('Matrice de confusion\n')
print(cm)

sb.set_theme(style="ticks", color_codes=True)

labels = pd.Series(y_predicted)
metier_pred = [y_m[y==i].unique()[0] for i in labels]
metier_pred = pd.Series(metier_pred)
data = pd.DataFrame({'index': x_ax, 'original': y_test, 'predicted': labels,
                     'Metie_pred':labels, 'Metie_original': y_test })  
cleanup_nums = {"Metie_pred": {i : y_lab[i]+'_pred' for i in range(len(y_lab))},
               "Metie_original": {i : y_lab[i]+'_original' for i in range(len(y_lab))}
               }
data.replace(cleanup_nums, inplace=True)
plt.figure(figsize=(15,10))
g =sb.scatterplot(y="Metie_pred", x="index",   data=data, hue="Metie_original")
#plt.yticks([0,1,2,3], y_lab)
plt.title("Classification par LogisticRegression\n\n")
plt.legend()
plt.show()


# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 4).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
cm = confusion_matrix(y_test, dtree_predictions) 
print('Precision de classification DescisionTreeClassifier: '+str(np.round(accuracy_score(y_test, dtree_predictions),2)*100)+'%   \n')
print(classification_report(y_test, dtree_predictions))
print('Matrice de confusion\n')
print(cm)

labels = pd.Series(dtree_predictions)
metier_pred = [y_m[y==i].unique()[0] for i in labels]
metier_pred = pd.Series(metier_pred)
data = pd.DataFrame({'index': x_ax, 'original': y_test , 'predicted': labels,
                     'Metie_pred':labels, 'Metie_original':y_test })  
cleanup_nums = {"Metie_pred": {i : y_lab[i]+'_pred' for i in range(len(y_lab))},
               "Metie_original": {i : y_lab[i]+'_original' for i in range(len(y_lab))}
               }
data.replace(cleanup_nums, inplace=True)
plt.figure(figsize=(15,10))
g =sb.scatterplot(y="Metie_pred", x="index",   data=data, hue="Metie_original")
plt.title("Classification par DescisionTreeClassifier\n\n")
plt.legend()
plt.show()

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")    
            
import xgboost as xgb

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error


clf_xgb = xgb.XGBClassifier(objective="multi:softprob", random_state=42)

clf_xgb.fit(X_train, y_train)

y_predicted = clf_xgb.predict(X_test)

print('Precision de classification par Gradient boosting est de: '+str(np.round(accuracy_score(y_test, y_predicted),2)*100)+'%   \n')
print(classification_report(y_test, y_predicted))
print('Matrice de confusion\n')
print(confusion_matrix(y_test, y_predicted))

labels = pd.Series(y_predicted)
metier_pred = [y_m[y==i].unique()[0] for i in labels]
metier_pred = pd.Series(metier_pred)
data = pd.DataFrame({'index': x_ax, 'original': y_test , 'predicted': labels,
                     'Metier_pred':labels, 'Metier_original': y_test })  
cleanup_nums = {"Metier_pred": {i : y_lab[i]+'_pred' for i in range(len(y_lab))},
               "Metier_original": {i : y_lab[i]+'_original' for i in range(len(y_lab))}
               }
data.replace(cleanup_nums, inplace=True)
plt.figure(figsize=(15,10))
sb.set_theme(style="ticks", color_codes=True)
g =sb.scatterplot(y="Metier_pred", x="index",   data=data, hue="Metier_original")
plt.title("Classification par XGBClassifier\n\n")
plt.legend()
plt.show()


metier_predicted= clf_xgb.predict(Metier_test)
metier_predicted
x_ax = range(len(metier_predicted))
labels = pd.Series(metier_predicted)
metier_pred = [y_m[y==i].unique()[0] for i in labels]
metier_pred = pd.Series(metier_pred)
data = pd.DataFrame({'index': x_ax, 'metier_manquants': labels,
                     'Metie_labels': labels })  
cleanup_nums = {"metier_manquants": {i : y_lab[i]+'_pred' for i in range(len(y_lab))}
               }
data.replace(cleanup_nums, inplace=True)
plt.figure(figsize=(15,10))
g =sb.scatterplot(y="metier_manquants", x="index",   data=data, hue='metier_manquants')
plt.title("Prédiction des métiers manquants par XGBClassifier \n\n")
plt.legend()
plt.show()
df.Metier.isnull().sum()
data.shape
data