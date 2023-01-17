from IPython.display import Image

Image("../input/pictures/airbnb.png")
import pandas as pds

import numpy as np

import matplotlib.pyplot as plt

import xgboost as xgb

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, recall_score,precision_score, f1_score, roc_curve

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split,cross_val_score, KFold, GridSearchCV, StratifiedKFold

import warnings

warnings.filterwarnings('ignore')
# =============================================================================#

# Etude du fichier age_gender.csv                                              #

# =============================================================================#

df_age = pds.read_csv("/kaggle/input/airbnb-data/age_gender_bkts.csv")

nRow, nCol = df_age.shape

print(f'Il y a {nRow} lignes and {nCol} colonnes dans le fichier age_gender_bkts.csv')

df_age.head(5)

#Regarder s'il y a des valeurs nulles dans le fichier

df_age.isnull().values.any() 

#Convertir le bucket 100+ en réel bucket : les buckets sont tous de taille 5

df_age['age_bucket'] = df_age['age_bucket'].apply(lambda x : '100-104' if x == '100+' else x)

df_age.head(5)

#Regarder quelles sont les differentes destinations, les differents genre et les differentes années

print(df_age['country_destination'].value_counts())

print(df_age['gender'].value_counts())

print(df_age['year'].value_counts())   #Il n'y a qu'une seule année : 2015 --> cette colonne ne sert à rien

df_age = df_age.drop('year', axis=1)     

df_age.head(5)
# =============================================================================#

# Etude du fichier countries.csv                                               #

# =============================================================================#

df_countries = pds.read_csv("/kaggle/input/airbnb-data/countries.csv")  

nRow, nCol = df_countries.shape

print(f'Il y a {nRow} lignes and {nCol} colonnes dans le fichier countries.csv')

df_countries.head(10)
#Regarder s'il y a des valeurs nulles dans le fichier

df_countries.isnull().values.any() 
# =============================================================================#

# Etude du fichier sessions.csv                                                #

# =============================================================================#

df_sessions = pds.read_csv("/kaggle/input/airbnb-data/sessions.csv")  

nRow, nCol = df_sessions.shape

print(f'Il y a {nRow} lignes and {nCol} colonnes dans le fichier countries.csv')

df_sessions.head(20)
sns.distplot(df_sessions[df_sessions['secs_elapsed'].notnull()]['secs_elapsed'])
df_sessions['secs_elapsed'].describe()
print(len(df_sessions[df_sessions['secs_elapsed'].isnull()]))

median_secs = df_sessions['secs_elapsed'].median()

df_sessions['secs_elapsed'] = df_sessions['secs_elapsed'].fillna(median_secs)

print(len(df_sessions[df_sessions['secs_elapsed'].isnull()]))

print(df_sessions['secs_elapsed'].describe())

df_sessions.head(5)
print(df_sessions['device_type'].value_counts())

df_countries.isnull().values.any()
# =============================================================================#

# Etude du fichier train_users_2.csv                                           #

# =============================================================================#

df_app = pds.read_csv("/kaggle/input/airbnb-data/train_users_2.csv") 

df_app['type_dataset'] = "apprentissage"

nRow, nCol = df_app.shape

print(f'Il y a {nRow} lignes and {nCol} colonnes dans le fichier train_users_2.csv')

df_test = pds.read_csv("/kaggle/input/airbnb-data/test_users.csv") 

df_test['type_dataset'] = "test"

df_test['country_destination'] ="predict"

df_all = pds.concat([df_app,df_test],ignore_index=True,sort=False)

df_app.head(5)
# =============================================================================#

# Nettoyage des données                                                        #

# =============================================================================#

#Transformation des date_first_booking

df_all['date_first_booking'].fillna('1800-01-01',inplace=True)

df_all['month_first_booking'] = df_all['date_first_booking'].apply(lambda x : int(x.split('-')[1]))

df_all['day_first_booking'] = df_all['date_first_booking'].apply(lambda x : int(x.split('-')[2]))

df_all['year_first_booking'] = df_all['date_first_booking'].apply(lambda x : int(x.split('-')[0]))

df_all.drop(columns=['date_first_booking'],axis=1, inplace=True)



#Transformation des date_account_created

df_all['date_account_created'].fillna('1800-01-01',inplace=True)

df_all['month_account_created'] = df_all['date_account_created'].apply(lambda x : int(x.split('-')[1]))

df_all['day_account_created'] = df_all['date_account_created'].apply(lambda x : int(x.split('-')[2]))

df_all['year_account_created'] = df_all['date_account_created'].apply(lambda x : int(x.split('-')[0]))

df_all.drop(columns=['date_account_created'],axis=1, inplace=True)



#Transformation des timestamp

df_all['timestamp_first_active'] = df_all['timestamp_first_active'].apply(lambda x : int(str(x)[0:8]))

df_all['year_first_active'] = df_all['timestamp_first_active'].apply(lambda x : int(str(x)[0:4]))

df_all['month_first_active'] = df_all['timestamp_first_active'].apply(lambda x : int(str(x)[4:6]))

df_all['day_first_active'] = df_all['timestamp_first_active'].apply(lambda x : int(str(x)[6:8]))



#Remplacer les nan dans 'first_affiliate_tracked' par 'None'

df_all['first_affiliate_tracked'].replace(np.nan,'none', inplace=True)

df_all['country_destination'].replace(np.nan,'other', inplace=True)



#Joindre avec countries pour obtenir la colonne language_levenshtein_distance

df_countries['language'] = df_countries['destination_language'].apply(lambda x : x[0:2])

join_countries = df_countries[['language','language_levenshtein_distance']].drop_duplicates()

df_all = df_all.merge(join_countries,on='language',how='left')



#df_all = df_all.merge(df_sessions, left_on="id", right_on="user_id", how='outer')



#Nettoyage des colonnes rajoutées à partir de sessions

#df_all['action_type'].replace(np.nan,'unknown', inplace=True)

#df_all['action_type'].replace('-unknown-','unknown', inplace=True)

#df_all['device_type'].replace(np.nan,'unknown', inplace=True)

#df_all['device_type'].replace('-unknown-','unknown', inplace=True)

df_all['gender'].replace('-unknown-',np.nan, inplace=True)

df_all['first_browser'] = df_all['first_browser'].replace('-unknown-', np.nan)

df_all.head(5)
#print(df_all['age'].value_counts())

df_all[df_all['age'] > 100].head()
from datetime import date

df_all['age'] = df_all['age'].apply(lambda age : date.today().year - age if (age > 100 and age < 2005) else (np.nan if age >= 2005 else age))

df_all[df_all['age'] > 100].head()

print(df_all['age'].describe())
print(df_all['country_destination'].value_counts())
df_all = df_all[(df_all['country_destination'] != 'NDF') & (df_all['country_destination'] != 'other')]

print(df_all['country_destination'].value_counts())                       
df_app = df_all[df_all['type_dataset']=="apprentissage"]

df_test = df_all[df_all['type_dataset']=="test"]

df_app = df_app.drop('type_dataset', axis=1)

classes = ['US','FR','IT','GB','ES','CA','DE','NL','AU','PT']



def stacked_bar(feature,size=(6, 6)):

    ctab = pds.crosstab([df_app[feature].fillna('Unknown')], df_app.country_destination, dropna=False).apply(lambda x: x/x.sum(), axis=1)

    ctab[classes].plot(kind='bar', stacked=True, colormap='gist_ncar', legend=True,figsize=size)

stacked_bar('gender')
sns.distplot(df_app['age'].dropna())
def age_group(x) : 

    if x <= 20 :

        return 'Teenager'

    elif 20 <= x and x < 40 :

        return 'Young'

    elif 40 <= x and x < 60 :

        return 'Adult'

    elif 60 <= x and x < 100: 

        return 'Old '

    else : 

        return 'Unknown'

    

df_app['age_group'] = df_app['age'].apply(lambda age : age_group(age))   

stacked_bar('age_group')
df_app = df_app.drop('age_group', axis=1)
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))

sns.boxplot(x='country_destination', y='age', data=df_app, palette="muted", ax =ax)

ax.set_ylim([10, 75])
stacked_bar('language', size=(15,6))

stacked_bar('signup_app')

stacked_bar('first_device_type')

stacked_bar('first_browser', size=(15,6))
stacked_bar('signup_method')
stacked_bar('affiliate_channel')

stacked_bar('affiliate_provider')

stacked_bar('first_affiliate_tracked')
class_dict = {

    'US': 0,

    'FR': 1,

    'CA': 2,

    'GB': 3,

    'ES': 4,

    'IT': 5,

    'PT': 6,

    'NL': 7,

    'DE': 8,

    'AU': 9

}

X, y = df_app.drop('country_destination', axis=1), df_app['country_destination'].apply(lambda x: class_dict[x])

print("ok")
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

tolabelize_columns = ['gender','signup_method','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser'] 

def labelize(df) : 

  for col in tolabelize_columns :

    df[col]=le.fit_transform(df[col].astype(str))

  return df



X = labelize(X)

X = X.drop(columns=['id'])

X.fillna(-1, inplace=True)

X.isnull().values.any()
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, stratify=y)

print(len(train_X))

print(len(test_X))

print(len(train_y))

print(len(test_y))
print(train_y.value_counts())
print(test_y.value_counts())
Image("../input/pictures/matrice_confusion.png")
# =============================================================================#

# Application des forets aléatoires                                            #

# =============================================================================#

from sklearn.metrics import confusion_matrix

rfc1 = RandomForestClassifier()

    #Apprendre aux modèles nos données

rfc1.fit(train_X, train_y)

    #Vérifier que notre score sur nos données de validation n'est ni trop faible (underfitting) ni trop fort (overfitting)

        #1) Prédire nos données d'apprentissage (sur lesquelles nous avons appris à notre modèle)

predictions_app = rfc1.predict(train_X)

        #2) Calculer son score de prediction

score = accuracy_score(train_y, predictions_app)

print("Score sur les données d'apprentissage : ", score*100 , "%")

    #Faire des prédictions en utilisant nos données de validation

predictions = rfc1.predict(test_X)

#print(np.unique(predictions))

    #Score

score = accuracy_score(test_y, predictions)

print("Score sur les données de test par random forest classifier : ", score*100 , "%")



    #Recall 

recall_score_micro = recall_score(test_y, predictions, average='micro')

recall_score_macro = recall_score(test_y, predictions, average='macro')

recall_score_weighted = recall_score(test_y, predictions, average='weighted')

print("recall_score_micro : ", recall_score_micro*100, "%")

print("recall_score_macro : ", recall_score_macro*100, "%")

print("recall_score_weighted : ", recall_score_weighted*100, "%")

    #Precision

precision_score_micro = precision_score(test_y, predictions, average='micro')

precision_score_macro = precision_score(test_y, predictions, average='macro')

precision_score_weighted = precision_score(test_y, predictions, average='weighted')

print("precision_score_micro : ", precision_score_micro*100, "%")

print("precision_score_macro : ", precision_score_macro*100, "%")

print("precision_score_weighted : ", precision_score_weighted*100, "%")

    #FMesure

fmesure_score_micro = f1_score(test_y, predictions, average='micro')

fmesure_score_macro = f1_score(test_y, predictions, average='macro')

fmesure_score_weighted = f1_score(test_y, predictions, average='weighted')

print("fmesure_score_micro : ", fmesure_score_micro*100, "%")

print("fmesure_score_macro : ", fmesure_score_macro*100, "%")

print("fmesure_score_weighted : ", fmesure_score_weighted*100, "%")

            #Matrice de confusion 

df_int = df_app[(df_app['country_destination'] != 'predict')]

conf = confusion_matrix(test_y, predictions)

cf = pds.DataFrame(conf, columns=['prédict ' + cl for cl in df_int['country_destination'].drop_duplicates()])

cf.index = ['actual ' + cl for cl in df_int['country_destination'].drop_duplicates()]

print(cf)
# =============================================================================#

# Application des forets aléatoires avec max_depth = 15                        #

# =============================================================================#

rfc2 = RandomForestClassifier(max_depth=15)

    #Apprendre aux modèles nos données

rfc2.fit(train_X, train_y)

    #Vérifier que notre score sur nos données de validation n'est ni trop faible (underfitting) ni trop fort (overfitting)

        #1) Prédire nos données d'apprentissage (sur lesquelles nous avons appris à notre modèle)

predictions_app = rfc2.predict(train_X)

        #2) Calculer son score de prediction

score = accuracy_score(train_y, predictions_app)

print("Score sur les données d'apprentissage : ", score*100 , "%")

    #Faire des prédictions en utilisant nos données de validation

predictions = rfc2.predict(test_X)

print(np.unique(predictions))

    #Score

score = accuracy_score(test_y, predictions)

print("Score sur les données de test par random forest classifier : ", score*100 , "%")



    #Recall 

recall_score_weighted = recall_score(test_y, predictions, average='weighted')

print("recall_score_weighted : ", recall_score_weighted*100, "%")

    #Precision

precision_score_weighted = precision_score(test_y, predictions, average='weighted')

print("precision_score_weighted : ", precision_score_weighted*100, "%")

    #FMesure

fmesure_score_weighted = f1_score(test_y, predictions, average='weighted')

print("fmesure_score_weighted : ", fmesure_score_weighted*100, "%")

    #Matrice de confusion 

df_int = df_app[(df_app['country_destination'] != 'predict')]

conf = confusion_matrix(test_y, predictions)

cf = pds.DataFrame(conf, columns=['prédict ' + cl for cl in df_int['country_destination'].drop_duplicates()])

cf.index = ['actual ' + cl for cl in df_int['country_destination'].drop_duplicates()]

print(cf)
# =============================================================================#

# Application des forets aléatoires avec max_depth=15, n_estimators=20         #

# =============================================================================#

rfc3 = RandomForestClassifier(max_depth=15, n_estimators=20)

    #Apprendre aux modèles nos données

rfc3.fit(train_X, train_y)

    #Vérifier que notre score sur nos données de validation n'est ni trop faible (underfitting) ni trop fort (overfitting)

        #1) Prédire nos données d'apprentissage (sur lesquelles nous avons appris à notre modèle)

predictions_app = rfc3.predict(train_X)

        #2) Calculer son score de prediction

score = accuracy_score(train_y, predictions_app)

print("Score sur les données d'apprentissage : ", score*100 , "%")

    #Faire des prédictions en utilisant nos données de validation

predictions = rfc3.predict(test_X)

print(np.unique(predictions))

    #Score

score = accuracy_score(test_y, predictions)

print("Score par random forest classifier : ", score*100 , "%")



    #Recall 

recall_score_weighted = recall_score(test_y, predictions, average='weighted')

print("recall_score_weighted : ", recall_score_weighted*100, "%")

    #Precision

precision_score_weighted = precision_score(test_y, predictions, average='weighted')

print("precision_score_weighted : ", precision_score_weighted*100, "%")

    #FMesure

fmesure_score_weighted = f1_score(test_y, predictions, average='weighted')

print("fmesure_score_weighted : ", fmesure_score_weighted*100, "%")

    #Matrice de confusion 

df_int = df_app[(df_app['country_destination'] != 'predict')]

conf = confusion_matrix(test_y, predictions)

cf = pds.DataFrame(conf, columns=['prédict ' + cl for cl in df_int['country_destination'].drop_duplicates()])

cf.index = ['actual ' + cl for cl in df_int['country_destination'].drop_duplicates()]

print(cf)
Image("../input/pictures/foret.png")
# =============================================================================#

# Création d'une fonction pour les arbres de décisions                         #

# Nous avons mis les valeurs qui sont codées par defaut                        #

# =============================================================================#

from sklearn import tree

def decision_tree(max_depth=None, criteron='gini', min_samples_split=2) : 

    if max_depth is not None : 

        clf = tree.DecisionTreeClassifier(max_depth = max_depth)

    elif criteron != 'gini' :

        clf = tree.DecisionTreeClassifier(criterion = criteron)

    elif min_samples_split != 2 :

        clf = tree.DecisionTreeClassifier(min_samples_split = min_samples_split)

    elif max_depth is not None and criteron != 'gini' :

        clf = tree.DecisionTreeClassifier(max_depth, criterion)

    elif max_depth is not None and min_samples_split != 2 :

        clf = tree.DecisionTreeClassifier(max_depth, min_samples_split)

    else :

        clf = tree.DecisionTreeClassifier()

        #Apprendre aux modèles nos données

    training = clf.fit(train_X, train_y)

        #Vérifier que notre score sur nos données de validation n'est ni trop faible (underfitting) ni trop fort (overfitting)

            #1) Prédire nos données d'apprentissage (sur lesquelles nous avons appris à notre modèle)

    predictions_app = clf.predict(train_X)

        #2) Calculer son score de prediction

    score_app = accuracy_score(train_y, predictions_app)

    print("Score sur les données d'apprentissage : ", score_app*100 , "%")

    #tree.plot_tree(training)



        #Faire des prédictions en utilisant nos données de validation

    predictions = clf.predict(test_X)

    #print(np.unique(predictions))

        #Score

    score = accuracy_score(test_y, predictions)

    print("Score sur les données prédites par arbre : ", score*100 , "%")

        #Recall 

    recall_score_weighted = recall_score(test_y, predictions, average='weighted')

    print("recall_score_weighted : ", recall_score_weighted*100, "%")

        #Precision

    precision_score_weighted = precision_score(test_y, predictions, average='weighted')

    print("precision_score_weighted : ", precision_score_weighted*100, "%")

        #FMesure

    fmesure_score_weighted = f1_score(test_y, predictions, average='weighted')

    print("fmesure_score_weighted : ", fmesure_score_weighted*100, "%")

                #Matrice de confusion 

    df_int = df_app[(df_app['country_destination'] != 'predict')]

    conf = confusion_matrix(test_y, predictions)

    cf = pds.DataFrame(conf, columns=['prédict ' + cl for cl in df_int['country_destination'].drop_duplicates()])

    cf.index = ['actual ' + cl for cl in df_int['country_destination'].drop_duplicates()]

    print(cf)

    
#sans parametre

clf1 = decision_tree()
#max_depth = 5

clf2 = decision_tree(max_depth=5)
#max_depth = 20

clf3 = decision_tree(max_depth=20)
#max_depth = 10

clf4 = decision_tree(max_depth=10)
#max_depth = 8

clf5 = decision_tree(max_depth=8)
#max_depth = 5 + critère = entropy

clf6 = decision_tree(max_depth=5, criteron='entropy')
# =============================================================================#

# Application d'arbre de décision avec un échantillon minimal égal à 100       #

# =============================================================================#

clf7 = decision_tree(min_samples_split=100)
# =============================================================================#

# Application d'arbre de décision avec max_depth=5,min_samples_split=100       #

# =============================================================================#

clf8 = decision_tree(max_depth=5,min_samples_split=100)
# =============================================================================#

# Récapitulatif des résultats obtenus                                          #

# =============================================================================#

Image("../input/pictures/arbre.png")
# =============================================================================#

# Application des bayesien naif                                                #

# =============================================================================#

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

    #Apprendre aux modèles nos données

gnb.fit(train_X, train_y)

    #Vérifier que notre score sur nos données de validation n'est ni trop faible (underfitting) ni trop fort (overfitting)

        #1) Prédire nos données d'apprentissage (sur lesquelles nous avons appris à notre modèle)

predictions_app = gnb.predict(train_X)

        #2) Calculer son score de prediction

score = accuracy_score(train_y, predictions_app)

print("Score sur les données d'apprentissage : ", score*100 , "%")

    #Faire des prédictions en utilisant nos données de validation

predictions = gnb.predict(test_X)

#print(np.unique(predictions))

    #Score

score = accuracy_score(test_y, predictions)

print("Score par bayesien naif classifier : ", score*100 , "%")



    #Recall 

recall_score_weighted = recall_score(test_y, predictions, average='weighted')

print("recall_score_weighted : ", recall_score_weighted*100, "%")

    #Precision

precision_score_weighted = precision_score(test_y, predictions, average='weighted')

print("precision_score_weighted : ", precision_score_weighted*100, "%")

    #FMesure

fmesure_score_weighted = f1_score(test_y, predictions, average='weighted')

print("fmesure_score_weighted : ", fmesure_score_weighted*100, "%")

        #Matrice de confusion 

df_int = df_app[(df_app['country_destination'] != 'predict')]

conf = confusion_matrix(test_y, predictions)

cf = pds.DataFrame(conf, columns=['prédict ' + cl for cl in df_int['country_destination'].drop_duplicates()])

cf.index = ['actual ' + cl for cl in df_int['country_destination'].drop_duplicates()]

print(cf)
# =============================================================================#

# Gradient Boosting                                                            #

# =============================================================================#

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier()

clf.fit(train_X,train_y)

    #Vérifier que notre score sur nos données de validation n'est ni trop faible (underfitting) ni trop fort (overfitting)

        #1) Prédire nos données d'apprentissage (sur lesquelles nous avons appris à notre modèle)

predictions_app = clf.predict(train_X)

        #2) Calculer son score de prediction

score = accuracy_score(train_y, predictions_app)

print("Score sur les données d'apprentissage : ", score*100 , "%")

    #Faire des prédictions en utilisant nos données de validation

predictions = clf.predict(test_X)

#print(np.unique(predictions))

    #Score

score = accuracy_score(test_y, predictions)

print("Score par gradient boosting classifier : ", score*100 , "%")



    #Recall 

recall_score_weighted = recall_score(test_y, predictions, average='weighted')

print("recall_score_weighted : ", recall_score_weighted*100, "%")

    #Precision

precision_score_weighted = precision_score(test_y, predictions, average='weighted')

print("precision_score_weighted : ", precision_score_weighted*100, "%")

    #FMesure

fmesure_score_weighted = f1_score(test_y, predictions, average='weighted')

print("fmesure_score_weighted : ", fmesure_score_weighted*100, "%")

        #Matrice de confusion 

df_int = df_app[(df_app['country_destination'] != 'predict')]

conf = confusion_matrix(test_y, predictions)

cf = pds.DataFrame(conf, columns=['prédict ' + cl for cl in df_int['country_destination'].drop_duplicates()])

cf.index = ['actual ' + cl for cl in df_int['country_destination'].drop_duplicates()]

print(cf)
# =============================================================================#

# Gradient Boosting                                                            #

# =============================================================================#

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=20)

clf.fit(train_X,train_y)

    #Vérifier que notre score sur nos données de validation n'est ni trop faible (underfitting) ni trop fort (overfitting)

        #1) Prédire nos données d'apprentissage (sur lesquelles nous avons appris à notre modèle)

predictions_app = clf.predict(train_X)

        #2) Calculer son score de prediction

score = accuracy_score(train_y, predictions_app)

print("Score sur les données d'apprentissage : ", score*100 , "%")

    #Faire des prédictions en utilisant nos données de validation

predictions = clf.predict(test_X)

#print(np.unique(predictions))

    #Score

score = accuracy_score(test_y, predictions)

print("Score par gradient boosting classifier : ", score*100 , "%")



    #Recall 

recall_score_weighted = recall_score(test_y, predictions, average='weighted')

print("recall_score_weighted : ", recall_score_weighted*100, "%")

    #Precision

precision_score_weighted = precision_score(test_y, predictions, average='weighted')

print("precision_score_weighted : ", precision_score_weighted*100, "%")

    #FMesure

fmesure_score_weighted = f1_score(test_y, predictions, average='weighted')

print("fmesure_score_weighted : ", fmesure_score_weighted*100, "%")

        #Matrice de confusion 

df_int = df_app[(df_app['country_destination'] != 'predict')]

conf = confusion_matrix(test_y, predictions)

cf = pds.DataFrame(conf, columns=['prédict ' + cl for cl in df_int['country_destination'].drop_duplicates()])

cf.index = ['actual ' + cl for cl in df_int['country_destination'].drop_duplicates()]

print(cf)
# =============================================================================#

# XG boost                                                                     #

# =============================================================================#

xgboost = XGBClassifier()

xgboost.fit(train_X,train_y)

    #Vérifier que notre score sur nos données de validation n'est ni trop faible (underfitting) ni trop fort (overfitting)

        #1) Prédire nos données d'apprentissage (sur lesquelles nous avons appris à notre modèle)

predictions_app = xgboost.predict(train_X)

        #2) Calculer son score de prediction

score = accuracy_score(train_y, predictions_app)

print("Score sur les données d'apprentissage : ", score*100 , "%")

    #Faire des prédictions en utilisant nos données de validation

predictions = xgboost.predict(test_X)

#print(np.unique(predictions))

    #Score

score = accuracy_score(test_y, predictions)

print("Score sur les données de validation par gradient boosting classifier : ", score*100 , "%")



    #Recall 

recall_score_weighted = recall_score(test_y, predictions, average='weighted')

print("recall_score_weighted : ", recall_score_weighted*100, "%")

    #Precision

precision_score_weighted = precision_score(test_y, predictions, average='weighted')

print("precision_score_weighted : ", precision_score_weighted*100, "%")

    #FMesure

fmesure_score_weighted = f1_score(test_y, predictions, average='weighted')

print("fmesure_score_weighted : ", fmesure_score_weighted*100, "%")

    #Matrice de confusion 

df_int = df_app[(df_app['country_destination'] != 'predict')]

conf = confusion_matrix(test_y, predictions)

cf = pds.DataFrame(conf, columns=['prédict ' + cl for cl in df_int['country_destination'].drop_duplicates()])

cf.index = ['actual ' + cl for cl in df_int['country_destination'].drop_duplicates()]

print(cf)
df_test = df_all[df_all['type_dataset']=="test"]

id = df_test['id']

df_test.head(5)

df_test = df_test.drop('country_destination', axis=1)

df_test = df_test.drop('type_dataset', axis=1)

df_test = df_test.drop('id', axis=1)

tolabelize_columns = ['gender','signup_method','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser'] 

def get_original_country(x) :

    for c in class_dict : 

        if class_dict[c] == x : 

            return c

        

def labelize(df) : 

  for col in tolabelize_columns :

    df[col]=le.fit_transform(df[col].astype(str))

  return df



df_test = labelize(df_test)

predictions = xgboost.predict(df_test)

result = pds.DataFrame(data={'country' : predictions, 'id' : id})

result['country'] = result['country'].apply(lambda x : get_original_country(x))

print(result.head(10))

print(np.unique(result['country']))

result.to_csv("soumission_projet_xgboost.csv", index=False)