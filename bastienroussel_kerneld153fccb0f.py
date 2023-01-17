import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
sns.set()
df = pd.read_csv('../input/data_v1.0.csv', sep=',')
df.head()
df.info()
print(df[df['Unnamed: 0'] != df['index']])
print(np.where(df['index'].values-np.arange(0, 20000, 1)!=0))
df = df.drop(['Unnamed: 0', 'index'], axis=1)
df.head()
print('Nbr de lignes incomplètes :', len(df.loc[df.date.notnull()&df.cheveux.notnull()&df.age.notnull()&df.exp.notnull()&df.salaire.notnull()&df.sexe.notnull()&df.diplome.notnull()&df.specialite.notnull()&df.note.notnull()&df.dispo.notnull()]))
df_intermediate = df.loc[df.date.notnull()&df.cheveux.notnull()&df.age.notnull()&df.exp.notnull()&df.salaire.notnull()&df.sexe.notnull()&df.diplome.notnull()&df.specialite.notnull()&df.note.notnull()&df.dispo.notnull()]
print('Perte informations lié à la projection : {:.2%}'.format(1-19021/20000))
print('Ratio embauche/candidature sur les 20 000 instances : {:.2%}'.format(len(df[df['embauche']==1].embauche.values)/len(df.embauche.values)))
print('Ratio emabuche/candidature sur les 19 021 instances complètes : {:.2%}'.format(len(df_intermediate[df_intermediate['embauche']==1].embauche.values)/len(df_intermediate.embauche.values)))
df_intermediate.describe()
print('Nbr instances possédants une expérience < 0 : {}'.format(len(df_intermediate[df_intermediate.exp < 0])))
print('Nbr instances possédants un age < 0 : {}'.format(len(df_intermediate[df_intermediate.age < 0])))
print('Pourcentage instances embauchées avec un age < 16 ans : {:.2%}'.format(len(df_intermediate[(df_intermediate.embauche==1)&(df_intermediate.age<16)])/len(df_intermediate[df_intermediate['embauche']==1])))
print('Nbr instances avec un age <= exp : {}'.format(len(df_intermediate[df_intermediate.age <= df_intermediate.exp])))
print('Nbr instances avec un age <= exp+16 : {}'.format(len(df_intermediate[df_intermediate.age<=df_intermediate.exp+16])))
df_intermediate = df_intermediate[df_intermediate['age']>=0]
df_intermediate = df_intermediate[df_intermediate['exp']>=0]
df_intermediate.describe()
df_intermediate.specialite.unique()
df_admis = df_intermediate[df_intermediate.embauche==1].groupby('specialite').date.agg('count')
df_total = df_intermediate.groupby('specialite').date.agg('count')

labels = ['archeologie', 'detective', 'forage', 'geologie']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
sizes_admis = [df_admis.loc['archeologie']/df_admis.sum(), df_admis.loc['detective']/df_admis.sum(), df_admis.loc['forage']/df_admis.sum(), df_admis.loc['geologie']/df_admis.sum()]
sizes_total = [df_total.loc['archeologie']/df_total.sum(), df_total.loc['detective']/df_total.sum(), df_total.loc['forage']/df_total.sum(), df_total.loc['geologie']/df_total.sum()]

plt.subplots_adjust(wspace=.6, hspace=.01)

plt.subplot(121)
plt.pie(sizes_admis, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Admis', fontsize=16)
plt.axis('equal')
plt.subplot(122)
plt.pie(sizes_total, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Candidats', fontsize=16)
plt.axis('equal')
plt.show()
np.array(sizes_total).sum()
def taux_succes(critere, df):
    print('\n')
    print('Taux de succès - {}'.format(critere))
    print('\n')
    for elt in df[str(critere)].unique():
        print('Taux de succès {} : {:.2%}'.format(elt, df[df[str(critere)]==elt].embauche.sum()/len(df[df[str(critere)]==elt])))
    print('-'*50)
df_intermediate.head()
taux_succes('specialite', df_intermediate)
taux_succes('diplome', df_intermediate)
taux_succes('cheveux', df_intermediate)
taux_succes('dispo', df_intermediate)
taux_succes('sexe', df_intermediate)
df_intermediate['annee'] = df_intermediate['date'].apply(lambda x: int(str(x)[:4]))
df_intermediate['mois'] = df_intermediate.apply(lambda x: int(str(x.date)[5:7]), axis=1)
df_date = df_intermediate[['annee', 'mois', 'embauche']].groupby(['annee', 'mois']).agg(['count', 'sum', 'mean'])
fg = plt.figure(figsize=(10, 10))
plt.subplots_adjust(wspace=.1, hspace=.6)
for i in range(5):
    l = i + 1
    m = 2010 + i
    ax = fg.add_subplot(5, 1, l)
    ax.plot(df_date.loc[2010].index, df_date.loc[m]['embauche']['mean'].values)
    ax.set_title('Taux de succès en '+str(m)+' par mois', fontsize=17)
plt.show()
df_intermediate[df_intermediate['embauche']==1][['age', 'exp', 'salaire', 'note']].describe()
df_intermediate[df_intermediate['embauche']==0][['age', 'exp', 'salaire', 'note']].describe()
df_intermediate['age_bin'] = pd.cut(df_intermediate.age, 5, labels=[0, 1, 2, 3, 4]).astype(int)
df_intermediate['exp_bin'] = pd.cut(df_intermediate.exp, 5, labels=[0, 1, 2, 3, 4]).astype(int)
df_intermediate['salaire_bin'] = pd.cut(df_intermediate.salaire, 5, labels=[0, 1, 2, 3, 4]).astype(int)
df_intermediate['note_bin'] = pd.cut(df_intermediate.note, 5, labels=[0, 1, 2, 3, 4]).astype(int)
fg = plt.figure(figsize=[35, 30])

ax1 = fg.add_subplot(321)
ax2 = fg.add_subplot(322)
ax3 = fg.add_subplot(323)
ax4 = fg.add_subplot(324)
ax5 = fg.add_subplot(325)
ax6 = fg.add_subplot(326)

plt.subplots_adjust(wspace=.2, hspace=.8)

df_intermediate.pivot_table(values='embauche', index=['specialite', 'diplome'], aggfunc='mean').plot.bar(ax=ax1, fontsize=20)
ax1.set_title('Taux de succès par Spécialité et Diplôme', fontsize=35)
df_intermediate.pivot_table(values='embauche', index=['specialite', 'exp_bin'], aggfunc='mean').plot.bar(ax=ax2, fontsize=20)
ax2.set_title('Taux de succès par Spécialité et Expérience', fontsize=35)
df_intermediate.pivot_table(values='embauche', index=['specialite', 'note_bin'], aggfunc='mean').plot.bar(ax=ax3, fontsize=20)
ax3.set_title('Taux de succès par Spécialité et Note', fontsize=35)
pd.pivot_table(data=df_intermediate, values='embauche', index=['specialite', 'salaire_bin'], aggfunc='mean').plot.bar(ax=ax4, fontsize=20)
ax4.set_title('Taux de succès par Spécialité et Salaire', fontsize=35)
df_intermediate.pivot_table(values='embauche', index=['specialite', 'sexe'], aggfunc='mean').plot.bar(ax=ax5, fontsize=20)
ax5.set_title('Taux de succès par Spécialité et Sexe', fontsize=35)
df_intermediate.pivot_table(values='embauche', index=['specialite', 'dispo'], aggfunc='mean').plot.bar(ax=ax6, fontsize=20)
ax6.set_title('Taux de succès par Spécialité et Disponibilité', fontsize=35)

plt.show()
df_total_date = df_intermediate[['mois', 'embauche', 'specialite']].groupby('mois').agg(['count', 'mean'])
df_total_date_spe = df_intermediate[['mois', 'embauche', 'specialite']].groupby(['specialite', 'mois']).agg(['count', 'mean'])

fg = plt.figure(figsize=(10, 10))
plt.subplots_adjust(wspace=.1, hspace=.5)

i = 0
for spe in df_intermediate.specialite.unique():
    i += 1
    plt.subplot(4, 1, i)
    plt.plot(df_total_date.index, df_total_date_spe.loc[str(spe)]['embauche']['mean'])
    plt.title('Taux de succès par mois - '+str(spe), fontsize=17)
    
plt.show()
def corr_quali_quali(X, Y, df):
    ''' Quantifie la corrélation entre deux grandeurs qualitatives'''

    list_X = df[X].unique()
    list_Y = df[Y].unique()
    
    k = len(list_X)
    l = len(list_Y)
    
    Z = df.columns[0]
    
    if k <= l:
        pass
    else:
        list_X, list_Y = list_Y, list_X
        k, l = l, k
        X, Y = Y, X
    
    
    MPC_X_Y = np.ones((k, l))
    i = 0
    j = 0
    for eltx in list_X :
        j = 0
        for elty in list_Y:
            MPC_X_Y[i, j] = df[df[Y]==elty].groupby(X).agg('count')[Z].loc[eltx]/df[df[Y]==elty].groupby(X).agg('count')[Z].sum()
            j += 1
        i +=1
        
        
    MPC_Y_X = np.ones((l, k))
    i = 0
    for elty in list_Y:
        j = 0
        for eltx in list_X:
            MPC_Y_X [i, j] = df[df[X]==eltx].groupby(Y).agg('count')[Z].loc[elty]/df[df[X]==eltx].groupby(Y).agg('count')[Z].sum()
            j += 1
        i += 1
    
    corr = sqrt(np.linalg.det(np.dot(MPC_X_Y, MPC_Y_X)))
    
    return corr   
def corr_quant_quali(X_quant, Y_quali, df):
    '''Quantifie la corrélation entre une variable quantitative et une variable qualitative'''
    
    moyennes = df[[X_quant, Y_quali]].groupby(Y_quali)[X_quant].agg('mean').values
    effectifs = df[[X_quant, Y_quali]].groupby(Y_quali)[X_quant].agg('count').values
    X = df[X_quant].values
    var_int = np.sum(effectifs * np.power(moyennes - np.mean(X), 2)) / len(X)  
    var_tot = np.var(df[X_quant].values)
    
    return var_int/var_tot   
print('Evaluation des dépendances statistiques :\n')
print('Corrélation entre la spécialité et le sexe : {:.4}'.format(corr_quali_quali('specialite', 'sexe', df_intermediate)))
print('Corrélation entre le salaire et les cheveux : {:.4}'.format(corr_quant_quali('salaire', 'cheveux', df_intermediate)))
print('Corrélation entre expérience et la note : {:.4}'.format(df_intermediate.corr().note.loc['exp']))
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
df_intermediate = df_intermediate.reset_index(drop=True)
df_intermediate.head()
label = LabelEncoder()
df_intermediate['cheveux'] = label.fit_transform(df_intermediate.cheveux)
df_intermediate['diplome'] = label.fit_transform(df_intermediate.diplome)
df_intermediate['specialite'] = label.fit_transform(df_intermediate.specialite)
df_intermediate['sexe'] = label.fit_transform(df_intermediate.sexe)
df_intermediate['dispo'] = label.fit_transform(df_intermediate.dispo)
df_intermediate['annee'] = label.fit_transform(df_intermediate.annee)
df_intermediate = df_intermediate.drop(['date'], axis=1)
df_intermediate.info()
X = df_intermediate.drop(['embauche'], axis=1)
Y = df_intermediate['embauche']
classifieurs = []
classifieurs.append(KNeighborsClassifier())
classifieurs.append(LinearDiscriminantAnalysis())
classifieurs.append(LogisticRegression(random_state=0))
classifieurs.append(LinearSVC(random_state=0))
classifieurs.append(SVC(random_state=0))
classifieurs.append(RandomForestClassifier(random_state=0))
classifieurs.append(ExtraTreesClassifier(random_state=0))
classifieurs.append(GradientBoostingClassifier(random_state=0))
classifieurs.append(MLPClassifier())

cv_resultats = []
for classifieur in classifieurs:
    cv_resultats.append(cross_val_score(classifieur, X, Y, scoring='roc_auc', cv=5, n_jobs=-1))
    
cv_moyennes = []
cv_ecarts_types = []

for cv_resultat in cv_resultats:
    cv_moyennes.append(cv_resultat.mean())
    cv_ecarts_types.append(cv_resultat.std())

cv_res = pd.DataFrame({'Algorithme': ['KNC', 'LDA', 'LG', 'LSVC', 'SVC', 'RFC', 'ETC', 'GBC', 'MLP'], 'CV_moyenne': cv_moyennes, 'CV_ecarts_types': cv_ecarts_types})
cv_res
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(X_train, Y_train)
Y_pred = GBC.predict_proba(X_test)
Y_cat = GBC.predict(X_test)

print(classification_report(Y_test, Y_cat))
print('-'*50)
print('\n')
print('Matrice de confusion')
print('\n')
print(confusion_matrix(Y_test, Y_cat))
Variable_GBC = pd.DataFrame({'Variable': X_train.columns, 'Importance': GBC.feature_importances_}).sort_values(by='Importance', ascending=False)
sns.barplot('Importance', 'Variable', data=Variable_GBC)
plt.title('Importance des variables dans le modèle GBC', fontsize=17)
plt.xlabel('Importance relative')
plt.show()
liste_1 = list(X_test.iloc[np.where(Y_test!=Y_cat)].index)
liste_2 = list(X_test.loc[Y_test==1].index)
liste_f = []

for l in liste_1:
    if l in liste_2:
        liste_f.append(l)

print('Répartition des erreurs par spécialité :\n')

print(X.iloc[liste_f].groupby('specialite').cheveux.agg('count'))
classifieurs_2 = []
classifieurs_2.append(RandomForestClassifier(random_state=0))
classifieurs_2.append(ExtraTreesClassifier(random_state=0))
classifieurs_2.append(GradientBoostingClassifier(random_state=0))

cv_resultats_recall = []
for classifieur in classifieurs_2:
    cv_resultats_recall.append(cross_val_score(classifieur, X, Y, scoring='recall', cv=5, n_jobs=-1))
    
cv_moyennes = []
cv_ecarts_types = []

for cv_resultat in cv_resultats_recall:
    cv_moyennes.append(cv_resultat.mean())
    cv_ecarts_types.append(cv_resultat.std())

cv_res_recall = pd.DataFrame({'Algorithme': ['RFC', 'ETC', 'GBC'], 'CV_moyenne': cv_moyennes, 'CV_ecarts_types': cv_ecarts_types})
cv_res_recall
ETC = ExtraTreesClassifier(random_state=0)
ETC.fit(X_train, Y_train)
Y_pred = ETC.predict_proba(X_test)
Y_cat = ETC.predict(X_test)

print(classification_report(Y_test, Y_cat))
print('-'*50)
print('\n')
print('Matrice de confusion')
print('\n')
print(confusion_matrix(Y_test, Y_cat))
Variable_ETC = pd.DataFrame({'Variable': X_train.columns, 'Importance': ETC.feature_importances_}).sort_values(by='Importance', ascending=False)
sns.barplot('Importance', 'Variable', data=Variable_ETC)
plt.title('Importance des variables dans le modèle ETC', fontsize=17)
plt.xlabel('Importance relative')
plt.show()
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

GBC_2 = GradientBoostingClassifier(random_state=0)

parameters = {'n_estimators': [50, 100, 150], 'min_samples_split': [30, 10], 'max_depth': [10, 20, 30], 
              'min_samples_leaf': [25, 15, 10], 'learning_rate': [0.1, 0.01]}

GS_GBC = GridSearchCV(GBC_2, param_grid=parameters, cv=3, scoring='recall', n_jobs=-1, verbose=1)
GS_GBC.fit(X_train, Y_train)
print('Best score :', GS_GBC.best_score_)
print('Best paramter set :', GS_GBC.best_estimator_.get_params())
GBC_2 = GradientBoostingClassifier(random_state=0, n_estimators=50, min_samples_split=10, 
                                   max_depth=30, min_samples_leaf=10, learning_rate=0.1)
GBC_2.fit(X_train, Y_train)
Y_pred = GBC_2.predict_proba(X_test)
Y_cat = GBC_2.predict(X_test)

print(classification_report(Y_test, Y_cat))
print('-'*50)
print('\n')
print('Matrice de confusion')
print('\n')
print(confusion_matrix(Y_test, Y_cat))
Variable_GBC_2 = pd.DataFrame({'Variable': X_train.columns, 'Importance': GBC_2.feature_importances_}).sort_values(by='Importance', ascending=False)
sns.barplot('Importance', 'Variable', data=Variable_GBC_2)
plt.title('Importance des variables dans le modéle GBC_2', fontsize=17)
plt.xlabel('Importance relative')
plt.show()
liste_1 = list(X_test.iloc[np.where(Y_test!=Y_cat)].index)
liste_2 = list(X_test.loc[Y_test==1].index)
liste_f = []

for l in liste_1:
    if l in liste_2:
        liste_f.append(l)

print('Répartition des erreurs par spécialité :\n')

print(X.iloc[liste_f].groupby('specialite').cheveux.agg('count'))