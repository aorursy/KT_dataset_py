import numpy as np 
import pandas as pd

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
df = pd.DataFrame()
df
# Création d'un DataFrame de 2 lignes sur 2 colonnes
dfa = pd.DataFrame(np.array([[10, 11], [20, 21]]))
dfa
# Création d'un tableau de 2 lignes sur 4 colonnes
dfa = pd.DataFrame([pd.Series(np.arange(0, 5)),pd.Series(np.arange(10, 15))])
dfa
data = [1,2,3,4,5]
df = pd.DataFrame(data)
df
data = [['Jean',10],['Arno',12],['Anne',13]]
df = pd.DataFrame(data,columns=['Prénom','Age'])
df
data = [['Jean',10],['Arno',12],['Anne',13]]
# On spécifie le type float pour les numériques
df = pd.DataFrame(data,columns=['Prénom','Age'],dtype=float)
df
data = {'Prénom':['Anne', 'Pierre', 'jack', 'Steven'],'Age':[65,31,23,48]}
df = pd.DataFrame(data)
df
# On rajoute un index
data = {'Prénom':['Anne', 'Pierre', 'jack', 'Steven'],'Age':[65,31,23,48]}
df = pd.DataFrame(data, index=['rang1','rang2','rang3','rang4'])
df
# un dictionnaire par ligne
# c n'est pas présent sur la 1° ligne d'où la valeur Nan par défaut
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
df
# Idem mais en passant un index
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data, index=['first', 'second'])
df
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data, index=['first', 'second'],columns=['a','b','col3'])
df
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
df
dfa.shape
dfa = pd.DataFrame(np.array([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]),columns=['col1', 'col2', 'col3', 'col4', 'col5'])
dfa
dfa.columns
# Pour connaître le nom d'une colonne en particulier
dfa.columns[3]
#Création d'un DataFrame complet
# On reprend notre ancien df
exam_data  = {'nom': ['Anne', 'Alex', 'catherine', 'Jean', 'Emillie', 'Michel', 'Matieu', 'Laura', 'Kevin', 'Jonas'],
        'note': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'qualification': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data , index=labels)

index = df.index
colonne = df.columns
data = df.values

# Liste des index
print(index)
# Liste des colonnes
print(colonne)
# Listes des valeurs
print(data)
# Vous pouvez aussi récupérer directement les infos de cette manière
print(index.values)
print(colonne.values)
# On utilise la fonction rename de Pandas
dfa.rename(columns={'col1':'colonne 1'}, inplace=True)
dfa.columns
# On peut aussi combiner avec un dictionnaire
old_names = ['colonne 1', 'col2'] 
new_names = ['new 1', 'colonne 2']
dfa.rename(columns=dict(zip(old_names, new_names)), inplace=True)
dfa
# Vous pouvez également utiliser une fonction plus poussée
dfa.columns = ['colonne 3' if x=='col3' else x for x in dfa.columns]
dfa
s1 = pd.Series(np.arange(1, 6, 1), index=[0,1,2,3,4])
s2 = pd.Series(np.arange(6, 11, 1))
dfa=pd.DataFrame({'c1': s1, 'c2': s2})
dfa

# Création d'un dataFrame ans index
dfa = pd.DataFrame({'month': [1, 4, 7, 10],
                  'year': [2012, 2014, 2013, 2014],
                  'sale':[55, 40, 84, 31]})
dfa
# Ajout d'un index depuis une colonne
dfa.set_index('sale')

dfa
import numpy as np 
import pandas as pd 

securities = pd.read_csv("../input/securities.csv",index_col='Ticker symbol', usecols=[0,1,2,3,6,7])
securities.head()

securities.tail()
# Quelle est la taille de mon DataFrame
len(securities)
#Examinons l'index
securities.index
# Nous pouvons examiner les autres colonnes du DataFrame
securities.columns
securities['Security']

# Vous pouvez aussi utiliser directement le nom de la colonne
print(securities.Security)

# Mais cela ne fonctionne pas si il y a un espace dans le nom de votre colonne

#Vous pouvez transformer votre colonne en un nouveau DataFrame
macolonne=securities.Security

mcdf=macolonne.to_frame()
# Nouveau DataFrame
print(mcdf)
# Pour info vous vaez accès à la completion
#securities.Security.

securities.columns.get_loc('CIK')
securities[:3]
# Vous pouvez également sélectionner avec le nom des index
securities['ABT':'ABBV']

securities[4:7]
# Pour sélectionner depuis un indice 
securities.iloc[1]
# Pour sélectionner depuis une valeur d'index
securities.loc['ABT']
securities['Security'][:3]
securities[:3]['Security']
# On sélectionne la cellule 2° ligne, 1° colonne
securities.at['ABT','Security']
# On sélectionne la cellule 2° ligne, 1° colonne
# Mais en passant des valeurs.
securities.iat[1,0]
# Sélection de toutes les lignes avec une valeur de colonne CIK > 15000
securities.CIK > 15000
securities[securities.CIK > 15000]
securities[(securities.CIK > 15000) & (securities.CIK<70000)][['CIK']]

# On va changer le nom d'une colonne et stocker le résultat dans un nouveau DataFrame
test = securities.rename(columns={'CIK':'C I K'})

test.head()
# On peut effectuer la même opération directement sur notre dataframe
securities.rename(columns={'CIK':'C I K'}, inplace=True)
securities.head()
copie = securities.copy()
# On ajoute une nouvelle colonne
copie['fois2'] = securities['C I K'] * 2
copie[:2]
# On réalise une copie de notre dataFrame initial
copie = securities.copy()

# On ajoute une colonne
copie.insert(1, 'fois2', securities['C I K'] * 2)
copie[:2]
# Extraction des 5 premières lignes avec juste la colonne C I K
rcopie = securities[0:5][['CIK']].copy()
rcopie
#Création d'une série
s = pd.Series( {'ABT': 'Cette action est présente','LOPMK': 'Cette action est absente'} )

s
# On va créer dynamiquement un nouvelle colonne commentaire et va fusionner avec la séries s
rcopie['Commentaire'] = s
rcopie
d = {'Un' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'Deux' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

dfexemple = pd.DataFrame(d)



print ("Ajout d'une nouvelle colonne en passant une série")
# Index c n'existe pas d'où valeur Nan
dfexemple['Trois']=pd.Series([10,20,30],index=['a','b','d'])
print(dfexemple)

print ("Ajout d'une nouvelle colonne à partir d'éléments existants")
dfexemple['Quatre']=dfexemple['Un']+dfexemple['Trois']
print(dfexemple)
# On remplace la colonne CIK
copie = securities.copy()
copie.CIK = securities.CIK * 2
copie[:5]
# On va créer une nouvelle colonne couleur
exam_data  = {'nom': ['Anne', 'Alex', 'catherine', 'Jean', 'Emillie', 'Michel', 'Matieu', 'Laura', 'Kevin', 'Jonas'],
        'note': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'qualification': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data , index=labels)

couleur = ['Rouge','Bleue','Orange','Rouge','Blanc','Blanc','Bleue','Vert','Vert','Rouge']
df['couleur'] = couleur
df
#Reprenons notre série
exam_data  = {'nom': ['Anne', 'Alex', 'catherine', 'Jean', 'Emillie', 'Michel', 'Matieu', 'Laura', 'Kevin', 'Jonas'],
        'note': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'qualification': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data , index=labels)
copie=df.copy()
del copie['note']
copie
copie=df.copy()
# On supprime la colonne note
popped = copie.pop('note')
# La colonne note a bien disparue
print(copie)
# On a récupéré le contenu de note
print(popped)
copie=df.copy()
# On supprime la colonne
afterdrop = copie.drop(['note'], axis = 1)
#Création du nouveau dataFrame
print('Contenu du résultat : ')
print(afterdrop)
print('contenu de copie inchangé')
print(copie)
# on supprimer 2 colonnes
copie=df.copy()
copie.drop(labels=['nom','qualification'], axis='columns').head()


# on peut aussi utiliser la commande del
d = {'Un' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'Deux' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

dfexemple = pd.DataFrame(d)

del dfexemple['Deux']
dfexemple
# On copie les 4° lignes
df1 = df.iloc[0:3].copy()
print(df1)
# copie des lignes 8,9 et 2
df2 = df.iloc[[8, 9, 2]]
print(df2)
# On ajout les lignes
ajout = df1.append(df2)
ajout
df3 = pd.DataFrame(0.0,index=df1.index,columns=['new'])
df3
df1.append(df3,sort=False)
df1
# On voit que la colonne New n'a pas été rajoutée
df1.append(df3,ignore_index=True, sort=False)
# On copie les 4° lignes
df1 = df.iloc[0:3].copy()

# copie des lignes 8,9 et 2
df2 = df.iloc[[8, 9, 2]]

pd.concat([df1, df2])
# copie de df2
df2_2 = df2.copy()
# Ajout d'un colonne
df2_2.insert(3, 'Foo', pd.Series(0, index=df2.index))
# On regarde ce que cela donne
df2_2
# On concatene
pd.concat([df1, df2_2])
r = pd.concat([df1, df2_2], keys=['df1', 'df2'])
r
df3 = df[:3][['nom', 'note']]
df3
df4 = df[:3] [['qualification']]
df4

#Concaténation sur l'axe
pd.concat([df3, df4], axis=1)
# On réalise une copie
df4_2 = df4.copy()
# On ajoute la colonne
df4_2.insert(1, 'note', pd.Series(1, index=df4_2.index))
df4_2
# On concatène
pd.concat([df3, df4_2], axis=1)
df5 = df[:3][['nom', 'note']]
df6 = df[2:5][['nom', 'note']]
pd.concat([df5, df6], join='inner', axis=1)
# Ajout d'une nouvelle ligne
a1 = df[:3].copy()
# Création d'une nouvelle ligne AVIS
# On assigne qq valeurs
a1.loc['z'] = ['New', 12.78, 'no']
a1
# Ajout d'une nouvelle colonne
a1 = df[:3].copy()
a1.loc[:,'AVIS'] = 0
a1
a1 = df[:3].copy()
apres_suppression = a1.drop(['b', 'c'])
print(apres_suppression)
# A1 n'a pas été modifié
print(a1)
# On cherche les lignes 
selection = df.note < 10
# On vérifie le nombre d'éléments
# ainsi que le nombre d'élements qui vérifie la règle
"{0} {1}".format(len(selection), selection.sum())
# On récupère les éléments qui ne correspondent pas à la règle
result = df[~selection]
result
# Les 3 premières lignes
troisPrem = df[:3]
troisPrem = df[:3].copy()
troisPrem
troisPrem = df[:3].copy()
troisPrem
troisPrem.loc['b', 'note'] = 0
troisPrem
troisPrem = df[:3].copy()
# On va recher l'identifant de la colonne
note_loc = df.columns.get_loc('note')
# On va cherche le n° d ela ligne 'b'
ligne_loc = df.index.get_loc('b')
# On change la valeur
troisPrem.iloc[ligne_loc, note_loc] = 15.5
troisPrem
# Initialisation des nombres aléatoire
np.random.seed(654321)

#Création d'un dt de 5 lignes sur 4 colonnes
df = pd.DataFrame(np.random.randn(5,4), columns=["A","B","C","D"])
df
# On va multiplier l'ensemble de valeur du DataFrame par 3
df*3
# sélectionnons la 1° ligne
p = df.iloc[0]

# On soustrait cette valeur
diff= df - p
diff
# On peut également faire l'inverse
diff= p - df
diff

# On récupère les colonnes 2 et 3
s2 = p[1:3]
s2['E'] = 0
df+s2
# On prend la ligne 1,2,3 et les colonnes B C
extrait = df[1:4][['B', 'C']]
extrait
df - extrait
# On récupère la colonne A
a_col = df['A']
df.sub(a_col, axis=0)
# On reprend notre ancien df
exam_data  = {'nom': ['Anne', 'Alex', 'catherine', 'Jean', 'Emillie', 'Michel', 'Matieu', 'Laura', 'Kevin', 'Jonas'],
        'note': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'qualification': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data , index=labels)
reset_df=df.reset_index()
reset_df
reset_df.set_index('nom')
sensemble = df[:4].copy()
reindex = sensemble.reindex(index=['a','b','z'])
reindex
sensemble.reindex(columns=['note','nom','Avis'])
exam_data  = {'nom': ['Anne', 'Alex', 'catherine', 'Jean', 'Emillie', 'Michel', 'Matieu', 'Laura', 'Kevin', 'Jonas'],
        'note': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'qualification': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data , index=labels)
reindex = df.reset_index()
multi_IH = reindex.set_index(['index','nom'])
multi_IH
# On peut voir le type d'index
type(multi_IH.index)
# Combien y a t il de niveau dans l'index ?
len(multi_IH.index.levels)
# On peut avoir les infos d'un index donné
multi_IH.index.levels[0]
# pour avoir les infos de lignes d'un index en particulier
print(multi_IH.xs('b'))
# Si on veut uniquement une ligne du second index
# Il faut spécifier le niveau
print(multi_IH.xs('Alex',level=1))
multi_IH.xs('Alex',level=1, drop_level=False)
multi_IH.xs('a').xs('Anne')
# Une autre syntax est possible
multi_IH.xs(('a','Anne'))
exam_data  = {'nom': ['Anne', 'Alex', 'catherine', 'Jean', 'Emillie', 'Michel', 'Matieu', 'Laura', 'Kevin', 'Jonas'],
        'note': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'avis': [np.nan, 14, 12.5, 5, 9, 18, 10.5, 10, 18, 15],
        'qualification': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df.T
df.axes
df.dtypes
print(df.empty)

# Avec un DataFrame vide
df2 = pd.DataFrame()
print(df2.empty)
df.ndim
df.shape
df.size
df.values
#5 1° lignes
print(df.head())
print("")
#2 premières lignes
print(df.head(2))
#5 dernières lignes
print(df.tail())
print("")
#2 dernières lignes
print(df.tail(2))
exam_data  = {'nom': ['Anne', 'Alex', 'catherine', 'Jean', 'Emillie', 'Michel', 'Matieu', 'Laura', 'Kevin', 'Jonas'],
        'note': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'avis': [np.nan, 14, 12.5, 5, 9, 18, 10.5, 10, 18, 15],
        'qualification': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data , index=labels)
# calcul de la moyenne de la colonne note
df[:]['note'].mean()
# Moyenne de toutes les colonnes numériques
df.mean()
# Moyenne de chaque lignes
df.mean(axis=1)
print(df.sum())
print("")
print("Somme uniquement de la colonne note")
print(df['note'].sum())

# Somme de chaque ligne
print(df.sum(1))
df.std()
df.var()
df.median()
# Minimum de la colonne note
df['note'].min()
# maximum d ela colonne note
df['note'].max()
# Pour avoir l'indice du min de la colonne note
df['note'].idxmin()

# Pour avoir l'indice du max de la colonne note
df['note'].idxmax()
# Somme cumulée de la colonne
df['note'].cumsum()
# Pour des colonnes numériques
df.describe()
#Pour avoir des stats sur les colonnes non numériques
df.describe(include=['object'])
#Pour avoir des stats sur les colonnes numériques
df.describe(include=['number'])
df['qualification'].describe()
# Pour des valeurs non numérique
print(df['qualification'].count())
print(df['qualification'].unique())
print(df['qualification'].value_counts())
# Pour des valeurs numeriques
print(df['note'].count())
print(df['note'].unique())
print(df['note'].value_counts())
df['nom'].value_counts()
#mais
df['nom'].value_counts(normalize=True)