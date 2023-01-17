'''Ignorer l'obsolescence et les avertissements futurs.'''
def suppress_warnings():
    import warnings
    warnings.filterwarnings('ignore', category = DeprecationWarning) 
    warnings.filterwarnings('ignore', category = FutureWarning) 

'''Import des modules.'''
import numpy as np               # Pour l'algèbre linéaire
import pandas as pd              # Pour la maipulation des data
import matplotlib.pyplot as plt  # Visualisation 2D
import seaborn as sns            
import missingno as mn           # Pour voir les valeurs manquantes
from scipy import stats          # Pour certaines stats
from sklearn.model_selection import train_test_split
import category_encoders as ce


from scipy.stats import norm, skew, kurtosis 

from IPython.core.display import HTML

''' Génératin de graphiques'''
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


from IPython.display import display
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)

'''Personnalisation de la visualisation'''
plt.style.use('bmh')                    # Utilisation du style  bmh's pour les graphs
sns.set_style({'axes.grid':False})      # Supprime les lignes

'''On peut utiliser le gras , souligné etc avec markedown'''
from IPython.display import Markdown
def bold(string):
    display(Markdown(string))
    
pd.options.display.max_rows = 150
'''Chargement du fichier csv
Ce reporter au notebook précédent concernant le type des variables

'''

column_types = {
    "place" : "int64",
"placeoptin" : "int64",
"rapport" : "float32",
"cr-reunion" : "int64",
"cr-num" : "int64",
"cr-nb_partants" : "int64",
"cr-Hippodrome" : "object",
"cr-Evt" : "int64",
"cr-autostart" : "int64",
"cr-corde" : "object",
"cr-etat du terrain" : "object",
"cr-distance" : "object",
"Numcheval" : "int64",
"CoteProbable" : "int64",
"Sexe" : "object",
"NbRencontre" : "float32",
"NbBattus" : "float32",
"SommeNbRencontre" : "float32",
"SommeNbBattus" : "float32",
"NumCourseJockey" : "float32",
"QteCourseJockeyJour" : "float32",
"SumCote5Race" : "float32",
"MoyenneCote5Race" : "float32",
"qteCoteCheval" : "float32",
"ETCote5Race" : "float32",
"SumPoids" : "int64",
"SumIdJockey" : "float32",
"SumidHippodrome" : "float32",
"SumIdDeferre" : "float32",
"SumDistance" : "float32",
"SumAllocation" : "float32",
"PositionIdJockey" : "float32",
"PositionidHippodrome" : "float32",
"PositionIdDeferre" : "float32",
"PositionDistance" : "float32",
"PositionAllocation" : "float32",
"pourcentIdJockey" : "float32",
"pourcentidHippodrome" : "float32",
"pourcentIdDeferre" : "float32",
"pourcentDistance" : "float32",
"pourcentAllocation" : "float32",
"SommeCourse" : "float32",
"nbjourdercourse" : "float32",
"numeroagespec" : "float32",
"numerodistancespec" : "float32",
"numerodeferrespec" : "float32",
"numerosexespec" : "float32",
"numeroagesexedistancespec" : "float32",
"idjockeynumcoursenbrcoursespec" : "float32",
"jockeyReussite" : "float32",
"jockeyReussitePlusPlace" : "float32",
"jockeyReussiteSpecialite" : "float32",
"jockeynReussiteDistance" : "float32",
"jockeyReussiteIdcheval" : "float32",
"jockeyReussiteDeferre" : "float32",
"jockeynReussiteSpecialiteDeferre" : "float32",
"jockeyReussiteDistanceDeferre" : "float32",
"entraineurReussite" : "float32",
"entraineurReussitePlusPlace" : "float32",
"entraineurReussiteSpecialite" : "float32",
"entraineurnReussiteDistance" : "float32",
"entraineurReussiteIdcheval" : "float32",
"chevalReussite" : "float32",
"chevalReussitePlusPlace" : "float32",
"chevalReussiteSpecialite" : "float32",
"chevalnReussiteDistance" : "float32",
"coteprobable" : "float32",
"gains" : "float32"
}

courses = pd.read_csv('../input/export-1-pt-utf.csv',dtype=column_types,parse_dates=['cr-Date'],infer_datetime_format=True,sep=",")


#courses= pd.read_csv('../input/export-1-pt-utf.csv')
courses = courses.drop(columns=['placeoptin','rapport'])

''' On divise notre jeu de données en deux parties'''
train, test = train_test_split(courses, test_size=0.33, random_state=42)


bold('**Visualisation des Data d\'entrainement:**')
display(train.head(2))

test = courses.drop(columns=['place'])
bold('**Visualisation des Data de test:**')
display(test.head(2))
'''Configuration du DataFrame'''
bold('**Configuration du DataFrame:**')
display(courses.shape)

'''Nom de variables (nom des colonnes)'''
bold('**Noms des variables:**')
display(courses.columns)
'''Type des données de nos variables.'''
bold('**Type des données de nos variables:**')
# On parcours les différentes colonnes
listeInt=""
listeFloat=""
listeObj=""
for col in courses.columns:
    if courses[col].dtype in ['float32', 'float64']:
        listeFloat+=col+','
    if courses[col].dtype in [ 'int32','int64']:
        listeInt +=col+","
    if courses[col].dtype in ['object']:
        listeObj+=col+','

display("Type flottant : "+listeFloat)
display("Type Int : "+listeInt)
display("Type Objet : "+listeObj)
display("Type Date : cr-Date")
'''Pour analyser les variables catégorielles, nous allons créer trois fonctions personnalisées.
Les deux premières fonctions affichent les étiquettes des barres en échelle absolue et relative respectivement.
Et la 3ème créée un dataframe de données absolues et relatives et génère également des courbes de fréquence abs et relative pour chaque variable.'''

''' #1.Fonction d'affichage de barres en échelle absolue.'''
def abs_bar_labels():
    font_size = 15
    plt.ylabel('Fréquence absolue', fontsize = font_size)
    plt.xticks(rotation = 0, fontsize = font_size)
    plt.yticks([])
    
    # Set individual bar lebels in absolute number
    for x in ax.patches:
        ax.annotate(x.get_height(), 
        (x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 7), 
        textcoords = 'offset points', fontsize = font_size, color = 'black')
    
'''#2.Fonction d'affichage de barres à l'échelle relative.'''
def pct_bar_labels():
    font_size = 15
    plt.ylabel('Fréquence relative (%)', fontsize = font_size)
    plt.xticks(rotation = 0, fontsize = font_size)
    plt.yticks([]) 
    
    # Etiquette personnalisée
    for x in ax1.patches:
        ax1.annotate(str(x.get_height()) + '%', 
        (x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 7), 
        textcoords = 'offset points', fontsize = font_size, color = 'black')
         
'''#3.Fonction pour créer un DataFrame avec les valeurs absolue et relative. Affiche le graph correspondant.'''
def freq_absolue_et_relative(variable, horizontal=False):
    global  ax, ax1 
    # Création du DataFrame
    absolute_frequency = variable.value_counts()
    relative_frequency = round(variable.value_counts(normalize = True)*100, 2)
    # On multiplie par 100 et on arronde à 2 décimales
    df = pd.DataFrame({'Fréquence absolue':absolute_frequency, 'Fréquence relative(%)':relative_frequency})
    print('Fréquence absolue et relative de ',variable.name,':')
    display(df)
    
    # Génération du graph.
    fig_size = (18,5)
    font_size = 15
    title_size = 18
    if horizontal:
        ax =  absolute_frequency.plot.barh(title = 'Fréquence absolue de %s' %variable.name, figsize = fig_size)
    else:
        ax =  absolute_frequency.plot.bar(title = 'Fréquence absolue de %s' %variable.name, figsize = fig_size)
    ax.title.set_size(title_size)
    abs_bar_labels()  # Displays bar labels in abs scale.
    plt.show()
    
    if horizontal:
        ax1 = relative_frequency.plot.barh(title = 'Fréquence relative de %s' %variable.name, figsize = fig_size)
    else:
        ax1 = relative_frequency.plot.bar(title = 'Fréquence relative de %s' %variable.name, figsize = fig_size)

    ax1.title.set_size(title_size)
    pct_bar_labels() # Affiche les étiquettes.
    plt.show()
'''Tracez et comptez le nombre de gagnants à l'échelle absolue et relative.'''
freq_absolue_et_relative(courses["place"])
'''Tracez et comptez le nombre de placé à l'échelle absolue et relative.'''
freq_absolue_et_relative(courses["Sexe"])
'''Tracez et comptez à l'échelle absolue et relative la variable cr-corde.'''
freq_absolue_et_relative(courses["cr-corde"])
'''Tracez et comptez à l'échelle absolue et relative la variable cr-autostart.'''
  
freq_absolue_et_relative(courses["cr-autostart"])
'''Tracez et comptez à l'échelle absolue et relative la variable cr-Evt.'''
  
freq_absolue_et_relative(courses["cr-Evt"])
'''Tracez et comptez à l'échelle absolue et relative la variable cr-etat du terrain.'''
  
freq_absolue_et_relative(courses["cr-etat du terrain"])
'''Tracez et comptez à l'échelle absolue et relative la variable cr-reunion.'''
  
freq_absolue_et_relative(courses["cr-reunion"])
'''Tracez et comptez à l'échelle absolue et relative la variable cr-num.'''
  
freq_absolue_et_relative(courses["cr-num"])
'''Tracez et comptez à l'échelle absolue et relative la variable cr-Hippodrome.'''
  
freq_absolue_et_relative(courses["cr-Hippodrome"])
# On va créer unDatafRame pour stocker les valeurs de chaque hippodrome
ar=courses['cr-Hippodrome'].value_counts(dropna=False)
dfhip = pd.DataFrame(ar)
'''On calcul la moyenne des hippodromes'''
print("Moyenne : "+str(round(dfhip['cr-Hippodrome'].mean(),2)) +" Mediane : "+str(dfhip['cr-Hippodrome'].median()))
# Fonction de convertion
def convert_hip(val):
    if dfhip.loc[val][0]>127:
        return val
    else:
        return "Autre"

courses['cr-HippodromeCat']=courses['cr-Hippodrome'].apply(convert_hip)

'''Nombre d'hippodromes différents '''
print("Nombre d'hippodromes différents : "+str(courses['cr-Hippodrome'].value_counts(dropna=False).count()))

'''Tracez et comptez à l'échelle absolue et relative la variable cr-HippodromeCat.'''
  
freq_absolue_et_relative(courses["cr-HippodromeCat"])
'''Tracez et comptez à l'échelle absolue et relative la variable cr-distance.'''
  
freq_absolue_et_relative(courses["cr-distance"])
def convert_distance(val):
    val=val.replace('mètres','')
    val=val.replace('.','')
    val=val.strip()
    return val


courses['cr-distanceNum']=courses['cr-distance'].apply(convert_distance)  
courses['cr-distanceNum']=pd.to_numeric(courses['cr-distanceNum'], downcast='integer')
courses['cr-corde'] = pd.Categorical(courses['cr-corde'])
courses['cr-etat du terrain'] = pd.Categorical(courses['cr-etat du terrain'])
courses['Sexe'] = pd.Categorical(courses['Sexe'])
courses['cr-reunion'] = pd.Categorical(courses['cr-reunion'])
courses['cr-num'] = pd.Categorical(courses['cr-num'])
courses["cr-Evt"] = pd.Categorical(courses["cr-Evt"])
courses["cr-autostart"] = pd.Categorical(courses["cr-autostart"])
courses['cr-HippodromeCat'] = pd.Categorical(courses['cr-HippodromeCat'])
courses['place'] = pd.Categorical(courses['place'])
courses.describe().transpose()
courses=courses.drop("SumPoids", axis=1)
def graph_unitaire(df,nom_colonne,proba):
    f, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
    sns.kdeplot(df[nom_colonne],ax = ax1,color ='blue',shade=True,
                label=("Skewness : %.2f"%(df[nom_colonne].skew()),
                       "Kurtosis: %.2f"%(df[nom_colonne].kurtosis())))
    ax1.set_xlabel(nom_colonne,color='black',fontsize=12)
    ax1.set_title(nom_colonne + ' Kdeplot',fontsize=14)
    ax1.axvline(df[nom_colonne].mean() , color ='g',linestyle = '--')
    ax1.legend(loc ='upper right',fontsize=12,ncol=2)
    
    sns.distplot(df[nom_colonne] , fit=norm,ax = ax2);
    ax2.set_xlabel(nom_colonne,color='black',fontsize=12)
    ax2.set_title(nom_colonne + ' distribution',fontsize=14)
    ax2.axvline(df[nom_colonne].mean() , color ='g',linestyle = '--')  
    (mu, sigma) = norm.fit(df[nom_colonne])
    ax2.legend(['Normal dist. ($\mu=$ {:.2f} et $\sigma=$ {:.2f} )'.format(mu, sigma)],loc ='upper right',fontsize=12,ncol=2)
    
    sns.despine()
    plt.show()
    
    if proba==True:
        graph_duo(df,nom_colonne)
        
    return(df[nom_colonne].skew(),df[nom_colonne].kurtosis())

def graph_duo(df,nom_colonne):
    
    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(courses[nom_colonne], plot=plt)
    plt.show()


            
       
''' On parcours les différentes colonnes '''
for col in courses.columns:
    ''' Uniquement les colonnes numériques '''
    if courses[col].dtype in ['float32', 'int32','int64', 'float64']:            
            display(HTML('<strong>Analyse de la variable '+col+'</strong>'))
            (skew1,kurto1)=graph_unitaire(courses,col, True)
            
columns = ['nom' ,'avant','apres']
ma_liste = []

''' On parcours les différentes colonnes '''
for col in courses.columns:
    ''' Uniquement les colonnes numériques '''
    if courses[col].dtype in ['float32', 'int32','int64', 'float64']:    
        ''' Si Skewness > Skewness alors on transforme la colonne en LOG '''
        if abs(courses[col].skew())>=0.75:              
            #courses["LOG"+col] = np.log1p(courses[col])
            my_dict = {'nom':col,'avant':courses[col].skew(),'apres':np.log1p(courses[col]).skew()}
            ma_liste.append(dict(my_dict))
            ''' On supprime la colonne initiale pour ne conserver que celle transformer '''
            #courses=courses.drop(col, axis=1)
                
compare = pd.DataFrame(ma_liste, columns=columns)
print(compare)
for col in ['qteCoteCheval', 'numeroagespec','jockeyReussiteDeferre']:
    courses["LOG"+col] = np.log1p(courses[col])
    ''' On supprime la colonne initiale pour ne conserver que celle transformer '''
    courses=courses.drop(col, axis=1)
courses['CoteProbable'].value_counts()
''' Fonction de convertion '''
def Vers_Nan(val):
    if val>0:
        return val
    else:
        return np.NaN

courses['CoteProbable']=courses['CoteProbable'].apply(Vers_Nan)
courses['CoteProbable'].value_counts()
import missingno as mn

mn.matrix(courses)
courses['jour_semaine']=courses['cr-Date'].dt.dayofweek

courses['jour_semaine'].describe()
courses['mois_annee']=courses['cr-Date'].dt.month
courses['mois_annee'].describe()

courses['semaine_annee']=courses['cr-Date'].dt.weekofyear
courses['semaine_annee'].describe()
courses['jour_mois']=courses['cr-Date'].dt.day
courses['jour_mois'].describe()

for col in ['jour_semaine','jour_mois','semaine_annee','mois_annee']:
    ''' Uniquement les nouvelles colonnes  '''              
    display(HTML('<strong>Analyse de la variable '+col+'</strong>'))
    (skew1,kurto1)=graph_unitaire(courses,col, True)
def plotBarCat(df,feature,target):
    
    x0 = df[df[target]==0][feature]
    x1 = df[df[target]==1][feature]
    

    trace1 = go.Histogram(
        x=x0,
        opacity=0.95,
          name = "Perdu",
        marker=dict(color='#FF7F0E')
    )
    trace2 = go.Histogram(
        x=x1,
        opacity=0.35,  
        name = "Gagné",
        marker=dict(color='#35FF23')
    )

    data = [trace1,trace2]
    layout = go.Layout(barmode='overlay',
                      title=feature,
                       yaxis=dict(title='Count'
        ))
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='overlaid histogram')
    

plotBarCat(courses,"jour_semaine",'place')
plotBarCat(courses,"mois_annee",'place')
plotBarCat(courses,"jour_mois",'place')
plotBarCat(courses,"semaine_annee",'place')
'''Créez une fonction pour compter le total des valeurs aberrantes. Et tracer des variables avec et sans valeurs aberrantes.'''
def outliers(variable):
    global filtered
    # calculer 1°, 3° quartiles .
    q1, q3 = variable.quantile(0.25), variable.quantile(0.75)
    iqr = q3 - q1
    
    # Calculer le seuil inférieur et le seuil supérieur pour les valeurs aberrantes
    l_fence, u_fence = q1 - 1.5*iqr , q3 + 1.5*iqr   # Toute valeur inférieure à l_fence et supérieure à u_fence est une valeur aberrante.
    
    # Observations that are outliers
    outliers = variable[(variable<l_fence) | (variable>u_fence)]
    print('Total des valeurs abarrantes', variable.name,':', outliers.count())
    
    # Supprime les valeurs aberrantes
    filtered = variable.drop(outliers.index, axis = 0)

    # Create subplots
    out_variables = [variable, filtered]
    out_titles = [' Distribution avec VA', ' Distribution sans VA']
    title_size = 25
    font_size = 18
    plt.figure(figsize = (25, 15))
    for ax, outlier, title in zip(range(1,3), out_variables, out_titles):
        plt.subplot(2, 1, ax)
        sns.boxplot(outlier).set_title('%s' %outlier.name + title, fontsize = title_size)
        plt.xticks(fontsize = font_size)
        plt.xlabel('%s' %outlier.name, fontsize = font_size)
outliers(courses['CoteProbable'])
''' calculer 1°, 3° quartiles '''
q1, q3 = courses["CoteProbable"].quantile(0.25), courses["CoteProbable"].quantile(0.75)
iqr = q3 - q1
    
''' Calculer le seuil inférieur et le seuil supérieur pour les valeurs aberrantes '''
l_fence, u_fence = q1 - 1.5*iqr , q3 + 1.5*iqr   # Toute valeur inférieure à l_fence et supérieure à u_fence est une valeur aberrante.
    
''' Observations that are outliers '''
outliers = courses["CoteProbable"][(courses["CoteProbable"]<l_fence) | (courses["CoteProbable"]>u_fence)]
#print(outliers)

''' Supprime les valeurs aberrantes '''
filtered = courses["CoteProbable"].drop(outliers.index, axis = 0)
moyenne=filtered.mean()
print("Moyenne : "+str(moyenne))



''' Fonction de convertion '''
def Vers_moyenne(val,l_fence2,u_fence2,moy):
    ''' On remplace les valeurs abérrantes par la moyenne'''
    if (val<l_fence2) | (val>u_fence2):
        return moy
    else:
        return val
    
print("Stats avant la modif")    
print(courses['CoteProbable'].describe().transpose())
courses['CoteProbable']=courses['CoteProbable'].apply(Vers_moyenne,l_fence2=l_fence, u_fence2=u_fence, moy=moyenne)

print("Stats après la modif") 
print(courses['CoteProbable'].describe().transpose())
"""Pour chaque variable comptons les valeurs manquantes"""
bold('**Valeurs manquantes pour chaque variables:**')
display(courses.isnull().sum())
"""Créez un boxplot pour visualiser les variables corrélées avec l'cote probable. Extrayez d'abord les variables qui nous intéressent."""

def correlationCote(df,liste):
    
    if len(liste) % 2 == 0:
        nbrow = int(len(liste)/2)
    else:
        nbrow = int(len(liste)/2)+1
    
    correlation = df.loc[:, liste]
    fig, axes = plt.subplots(nrows = nbrow, ncols = 2, figsize = (50,150))
    for ax, column in zip(axes.flatten(), correlation.columns):
        sns.boxplot(x = correlation[column], y =  df['CoteProbable'], ax = ax)
        ax.set_title(column, fontsize = 23)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 20)
        ax.set_ylabel('Cote Probable', fontsize = 20)
        ax.set_xlabel('')
    fig.suptitle('Variables associées', fontsize = 30)
    fig.tight_layout(rect = [0, 0.03, 1, 0.95])
    return correlation


#correlation= correlationCote(courses,["cr-reunion","cr-num","cr-nb_partants","cr-HippodromeCat","cr-Evt","cr-autostart","cr-corde","jour_mois","jour_semaine","Numcheval"])
""" Traçons une carte de corrélation pour voir quelle variable est fortement corrélée avec la coteprobable. 
Nous devons convertir la variable catégorielle en une carte thermique de corrélation numérique pour tracer la corrélation.
Convertissez donc les variables catégorielles en variables numériques."""
from sklearn.preprocessing import LabelEncoder

def graph_co(correlation, df, col):
    correlation = correlation.agg(LabelEncoder().fit_transform)
    correlation[col] = df[col] # Inserting CoteProbable in variable correlation.
    correlation = correlation.set_index(col).reset_index() # Move CoteProbable at index 0.

    '''Création du graphique'''
    plt.figure(figsize = (60,60))
    sns.set(font_scale=1.4)
    sns.heatmap(correlation.corr(), cmap ='BrBG', annot = True,linewidths=.5)
    plt.title('Variables correlées avec la cote probable', fontsize = 18)
    plt.show()
#graph_co(correlation, courses, 'CoteProbable')
liste = []
for col in courses.columns:
    if col.startswith('ARRONDI'):
        courses = courses.drop(columns=[col])
        
# On va arrondir après une décimale
for col in courses.columns:
    if courses[col].dtype in ['float32', 'float64']:
        #courses = courses.drop(columns=[col])
        if col not in ['CoteProbable','qteCoteCheval','numeroagespec','jockeyReussiteDeferre','coteprobable']:
            courses["ARRONDI"+col]=courses[col].apply(lambda x:round(x,1))
            liste.append("ARRONDI"+col)

    
#correlation=correlationCote(courses,liste)
#graph_co(correlation, courses, 'CoteProbable')
''' On va créer un nouveau df avec niquement les colonnes de liste'''
df = courses.loc[:, liste]
df['CoteProbable'] = courses['CoteProbable'] 

''' on créé un df de corrélation'''
dfcor=df.corr().transpose()


listecor = []

''' On parcours'''
for x in liste:   
    ''' Notre liste avec uniquement des corrélations supérieures à 0.15'''
    if abs(dfcor.loc[x, "CoteProbable"])>0.15:
        listecor.append(x)

listecor.append('cr-nb_partants')

print(listecor)
            
''' On impute la cote probable avec la médiane des colonnes'''
courses['CoteProbable']=courses['CoteProbable'].fillna(courses.groupby(listecor)['CoteProbable'].transform('median'))

''' On regarde si cela a bien fonctionné en comptant le nombre de Nan'''
courses['CoteProbable'].isnull().sum()
#def blurp:
#df1=courses.groupby('cr-nb_partants')['CoteProbable'].transform()

#courses["CoteProbable"] = courses.groupby('cr-nb_partants')['CoteProbable'].transform(lambda x: x.fillna(x.median()))


#bold('**Valeurs manquantes après imputation:**')
#display(courses.isnull().sum())


def combinliste(seq, k):
    p = []
    i, imax = 0, 2**len(seq)-1
    while i<=imax:
        s = []
        j, jmax = 0, len(seq)-1
        while j<=jmax:
            if (i>>j)&1==1:
                s.append(seq[j])
            j += 1
        if len(s)==k:
            courses['CoteProbable']=courses['CoteProbable'].fillna(courses.groupby(s)['CoteProbable'].transform('median'))
            #scomb = " ".join(s)
            print(s)
            print("  Reste Nan : "+str(courses['CoteProbable'].isnull().sum()))
            if courses['CoteProbable'].isnull().sum()==0:
                return True
        i += 1 
    return False
res=combinliste(listecor,11)
res=combinliste(listecor,10)
res=combinliste(listecor,9)
res=combinliste(listecor,8)
res=combinliste(listecor,7)
res=combinliste(listecor,6)
print(courses['CoteProbable'].describe().transpose())
"""
Séparons en deux notre jeu de données pour l'entrainement et le test.
Nous avons besoin de notre variable cible sans valeurs manquantes pour effectuer le test d'association avec les variables prédicteurs.
"""
df_train = courses.iloc[:9000, :]
df_test = courses.iloc[9000:, :]
df_test = df_test.drop(columns = ['place'], axis = 1)

'''#1.Créer une fonction qui crée un boxplot entre les variables catégorielles et numériques et calcule la corrélation bisériale.'''
def boxplot_and_correlation(cat,num):
    '''cat =  variable categorielle, and num = variable numérique.'''
    plt.figure(figsize = (18,7))
    title_size = 18
    font_size = 15
    ax = sns.boxplot(x = cat, y = num)
    
    # Changement de couleur
    box = ax.artists[0]
    box1 = ax.artists[1]
    
    # Change l'apparence
    box.set_facecolor('red')
    box1.set_facecolor('green')
    plt.title('Association entre Place & %s' %num.name, fontsize = title_size)
    plt.xlabel('%s' %cat.name, fontsize = font_size)
    plt.ylabel('%s' %num.name, fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.show()
    print('Corrélation entre', num.name, ' et ', cat.name,':', stats.pointbiserialr(num, cat))

'''#2.Créer une autre fonction pour calculer la moyenne lorsqu'elle est groupée par variable catégorielle. Et aussi tracer la moyenne groupée.'''
def nume_grouped_by_cat(num, cat):
    global ax
    font_size = 15
    title_size = 18
    grouped_by_cat = num.groupby(cat).mean().sort_values( ascending = False)
    grouped_by_cat.rename ({1:'Placé', 0:'Perdu'}, axis = 'rows', inplace = True) # Renaming index
    grouped_by_cat = round(grouped_by_cat, 2)
    ax = grouped_by_cat.plot.bar(figsize = (18,5)) 
    abs_bar_labels()
    plt.title('Moyenne %s ' %num.name + ' de Placés vs perdus', fontsize = title_size)
    plt.ylabel('Moyenne ' + '%s' %num.name, fontsize = font_size)
    plt.xlabel('%s' %cat.name, fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.show()

'''#3.Cette fonction trace l'histogramme de la variable numérique pour chaque classe de variable catégorielle..'''
def num_hist_by_cat(num,cat):
    font_size = 15
    title_size = 18
    plt.figure(figsize = (18,7))
    num[cat == 1].hist(color = ['g'], label = 'Placé', grid = False)
    num[cat == 0].hist(color = ['r'], label = 'Perdu', grid = False)
    plt.yticks([])
    plt.xticks(fontsize = font_size)
    plt.xlabel('%s' %num.name, fontsize = font_size)
    plt.title('%s ' %num.name + ' Distribution des Placés vs Perdus', fontsize = title_size)
    plt.legend()
    plt.show()
   
'''#4.Créer une fonction pour calculer l'anova entre variable numérique et variable catégorielle'''
def anova(num, cat):
    from scipy import stats
    grp_num_by_cat_1 = num[cat == 1] # Groupe notre variable numérique par variable catégorielle.
    grp_num_by_cat_0 = num[cat == 0] # Groupe notre variable numérique par variable catégorielle
    f_val, p_val = stats.f_oneway(grp_num_by_cat_1, grp_num_by_cat_0) # Calculate f statistics and p value
    print('Résultat Anova entre ' + num.name, ' & '+ cat.name, ':' , f_val, p_val)  
    
'''#5.Créer une autre fonction qui calcule le test de Tukey entre notre variable numérique et catégorielle.'''
def tukey_test(num, cat):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey = pairwise_tukeyhsd(endog = num,   # Data numérique
                             groups = cat,   # Data catégorielle
                             alpha = 0.05)   # Niveau significatif
    
    summary = tukey.summary()   # See test summary
    print("Résultat test Tukey entre " + num.name, ' & '+ cat.name, ':' )  
    display(summary)        
df_train["place"]=df_train["place"].astype("int32")
df_train["cr-num"]=df_train["cr-num"].astype("int32")

#'''Créez un boxplot pour visualiser la force d'association de Placé avec cr-num. Calcule également la corrélation bisériale.'''
boxplot_and_correlation(df_train["place"], df_train["cr-num"])
#'''Créez un boxplot pour visualiser la force d'association de Placé avec jour_mois. Calcule également la corrélation bisériale.'''
boxplot_and_correlation(df_train["place"], df_train["jour_mois"])
#'''Créez un boxplot pour visualiser la force d'association de Placé avec jour_mois. Calcule également la corrélation bisériale.'''
boxplot_and_correlation(df_train["place"], df_train["cr-distanceNum"])

'''#1. Créer une fonction qui calcule la fréquence absolue et relative de la variable Placé par une variable catégorielle. Et trace ensuite la fréquence absolue et relative de Placé par une variable catégorielle."
'''
def crosstab(cat, cat_target):
    '''cat = categorical variable, cat_target = our target categorical variable.'''
    global ax, ax1
    fig_size = (18, 5)
    title_size = 18
    font_size = 15
    cat_grouped_by_cat_target = pd.crosstab(index = cat, columns = cat_target)
    cat_grouped_by_cat_target.rename({0:'Perdu', 1:'Placé'}, axis = 'columns', inplace = True)  # Renomme les colonnes
    pct_cat_grouped_by_cat_target = round(pd.crosstab(index = cat, columns = cat_target, normalize = 'index')*100, 2)
    pct_cat_grouped_by_cat_target.rename({0:'Perdu(%)', 1:'Placé(%)'}, axis = 'columns', inplace = True)
    
    # Plot absolute frequency of Survived by a categorical variable
    ax =  cat_grouped_by_cat_target.plot.bar(color = ['r', 'g'], title = 'Nombre absolu de Placé et de perdu par %s' %cat.name, figsize = fig_size)
    ax.title.set_size(fontsize = title_size)
    abs_bar_labels()
    plt.xlabel(cat.name, fontsize = font_size)
    plt.show()
    
    # Plot relative frequrncy of Survived by a categorical variable
    ax1 = pct_cat_grouped_by_cat_target.plot.bar(color = ['r', 'g'], title = 'Pourcentage de Placé et de perdu par %s' %cat.name, figsize = fig_size)
    ax1.title.set_size(fontsize = title_size)
    pct_bar_labels()
    plt.xlabel(cat.name, fontsize = font_size)
    plt.show()
    
'''#2.Créez une fonction pour calculer le chi_square  entre une variable catégorielle et une variable catégorielle cible.'''
def chi_square(cat, cat_target):
    cat_grouped_by_cat_target = pd.crosstab(index = cat, columns = cat_target)
    test_result = stats.chi2_contingency (cat_grouped_by_cat_target)
    print('Résultat du test Chi Square entre placé & %s' %cat.name + ':')
    display(test_result)

'''#3.Calcul de Bonferroni-adjusté'''
def bonferroni_adjusted(cat, cat_target):
    dummies = pd.get_dummies(cat)
    for columns in dummies:
        crosstab = pd.crosstab(dummies[columns], cat_target)
        print(stats.chi2_contingency(crosstab))
    print('\nColonnes:', dummies.columns)
crosstab(df_train["Sexe"], df_train["place"])
chi_square(df_train["Sexe"], df_train["place"])
'''Calcule pour sexe (H,F,M) Placé.'''
bonferroni_adjusted(df_train["Sexe"], df_train["place"])
crosstab(df_train["cr-corde"], df_train["place"])
chi_square(df_train["cr-corde"], df_train["place"])
crosstab(df_train["cr-autostart"], df_train["place"])
chi_square(df_train["cr-autostart"], df_train["place"])
crosstab(df_train["cr-Evt"], df_train["place"])
chi_square(df_train["cr-Evt"], df_train["place"])
crosstab(df_train["cr-etat du terrain"], df_train["place"])
chi_square(df_train["cr-etat du terrain"], df_train["place"])
crosstab(df_train["cr-HippodromeCat"], df_train["place"])
chi_square(df_train["cr-HippodromeCat"], df_train["place"])
'''Calcule pour HippodromeCat Placé.'''
bonferroni_adjusted(df_train["cr-HippodromeCat"], df_train["place"])
'''Créez une fonction qui trace l'impact de 3 variables prédicteurs à la fois sur une variable cible.'''
def multivariate_analysis(cat1, cat2, cat3, cat_target,Type=1):
    if type==1:
        font_size = 15
    else:
        font_size = 45
    grouped = round(pd.crosstab(index = [cat1, cat2, cat3], columns = cat_target, normalize = 'index')*100, 2)
    grouped.rename({0:'Perdu%', 1:'Placé%'}, axis = 1, inplace = True)
    
    if type==1:
        grouped.plot.barh(color = ['r', 'g'], figsize = (18,5))
    else:
        grouped.plot.barh(color = ['r', 'g'], figsize = (100,500))
    
    
    plt.xlabel(cat1.name + ',' + cat2.name + ',' + cat3.name, fontsize = font_size)
    plt.ylabel('Fréquence relative (%)', fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.legend(loc = 'best')
    plt.show()


multivariate_analysis(df_train["cr-corde"], df_train["cr-autostart"], df_train["cr-Evt"], df_train["place"])
multivariate_analysis(df_train["cr-corde"], df_train["Sexe"], df_train["Numcheval"], df_train["place"],2)
multivariate_analysis(df_train["cr-etat du terrain"], df_train["cr-num"], df_train["Sexe"], df_train["place"],2)
courses.shape
courses.dtypes
courses.drop(["cr-Hippodrome", "cr-distance","cr-Date"], axis=1, inplace=True)

for col in courses.columns:
    if col.startswith('ARRONDI'):
        courses.drop(columns=[col], axis=1, inplace=True)
courses.shape

courses.select_dtypes(include=['category'])
''
listeCat=courses.select_dtypes(include=['category']).columns.tolist() 
listeCat
''' On supprime le libellé place'''
listeCat.pop(0)

''' On force en type object les category'''

for col in listeCat:
    courses[col]=courses[col].astype("object")
''' Création d'un nouveau dataframe dans lequel les colonnes de listeCat ont été supprimées '''
coursesBinary = courses.drop(listeCat, axis = 1)

'''Création de l'encodeur binaire '''
ce_binary = ce.BinaryEncoder(cols = listeCat)

''' Transformation des datas'''
ce_binary.fit_transform(courses, coursesBinary)
coursesBinary.head()