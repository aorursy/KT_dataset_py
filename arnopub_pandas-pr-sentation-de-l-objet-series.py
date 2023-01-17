# Importation de la librairie Panda
import pandas as pd
import numpy as np
# Options pour contrôler l'affichage des données
pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
# Création d'une série avec un seul élément
s1 = pd.Series(2)
s1
s1 = pd.Series(['a','b','c','d'])
s1
#Création d'une série avec un index
s1 = pd.Series(['a','b','c','d'],index=[10,11,12,13])
s1

# Création depuis un objet numpy
import numpy as np

data = np.array(['a','b','c','d'])
s1 = pd.Series(data)
s1
s1 = pd.Series(data,index=[10,11,12,13])
s1
# Vous pouvez aussi créer la série depuis un dictionnaire python
data = {'a' : 0., 'b' : 1., 'c' : 2.}
s1 = pd.Series(data)
s1
# Avec un index
data = {'a' : 0., 'b' : 1., 'c' : 2.}
s1 = pd.Series(data,index=['b','c','d','a'])
s1
# Création d'une série depuis un nombre
s1 = pd.Series(5, index=[0, 1, 2, 3])
s1
# Accéder à un élement de la série
s1 = pd.Series(['a','b','c','d'])
s1[0]

#avec erreur s1[9]

data = {'a' : 0., 'b' : 1., 'c' : 2.}
s1 = pd.Series(data,index=['b','c','d','a'])
s1['c'] 
s1[1]

# Vous pouvez aussi récupérer plusieurs élements en même temps

s1[['a','c','d']]
# La longueur d'une série peut-être connue en utilisant la fonction len

len(s1)
# Vous pouvez aussi utiliser la propriéta size de votre objet

s1.size
# La propriété shape retourn un tuple avec en 1° élément le nombre d'items de votre liste
s1.shape
# Pour compter le nombre de valeur non nulle
s1.count()
# Pour déterminer le nombre de valeur uniques
s1.unique()
# le compte de chacun des items uniques d'une Série peut être obtenu en utilisant value_counts()
s1.value_counts()
data = np.array(['a','b','c','d','e','f','g','h','i','j'])
s1 = pd.Series(data)
# Les 5 premières
s1.head()
# les 2 premières 
s1.head(n=2)
#Les 5 dernières
s1.tail()

# Les 2 dernières
s1.tail(n=2)
s1.take([0, 5, 7])
ind = np.array(['a','b','c','d','e','f','g','h','i','j'])
data = np.array([12,52,41,16,43,85,74,74,95,23])
s1 = pd.Series(data, index=ind)
s1['a']
#Accès à cette série à l'aide d'une valeur entière 
s1[2]
s1[['b', 'd']]
s1 = pd.Series([1, 2, 3], index=[10, 11, 12])

s1.loc[11]

# 1° série
s1 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

# 2° série
s2 = pd.Series([40, 30, 20, 10], index=['d', 'c', 'b', 'a'])

# Addiitonnnons les séries
s1 + s2
s1 * 3
s1 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

# 2° série
# La valeur de l'index e n'existe pas dans s1
s2 = pd.Series([40, 30, 20, 10,52], index=['d', 'c', 'b', 'a', 'e'])

s2+s1
s1 = pd.Series([1.0, 2.0, 3.0], index=['a', 'a', 'b'])
s2 = pd.Series([4.0, 5.0, 6.0], index=['a', 'a', 'c'])
s1 + s2
# Avec Numpy
nda = np.array([1, 2, 3, 4, 5])
nda.mean()
nda = np.array([1, 2, 3, 4, np.NaN])
nda.mean()
# Avec Pandas
s = pd.Series(nda)
s.mean()
s.mean(skipna=False)

# Quelles sont les lignes avec une valeur supérieure à 10
s = pd.Series(np.arange(0, 15))
s > 5
# Sélection des valeurs supérieures à 8
logicalResults = s > 8
s[logicalResults]
s[s > 8]
s[(s > 5) & (s < 8)]
(s >= 3).all()
# Supérieur à zéro
(s >= 0).all()
# Ai je des valeurs > à 3 ?
s[s >=3].any()
#Combien de valeur sont inférieures à 5 ?
(s < 5).sum()

# Une série de 10 items
s = pd.Series(np.random.randn(10))
s
# On affecte un index
s.index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
s
# Graine du générateur de nombre aléatoir
np.random.seed(123456)

# création de deux séries
s1 = pd.Series(np.random.randn(3))
s2 = pd.Series(np.random.randn(3))

#on concatène les 2 séries
combined = pd.concat([s1, s2])
combined
# réordonner l'index
combined.index = np.arange(0, len(combined))
combined
np.random.seed(987654)
s1 = pd.Series(np.random.randn(4), ['a', 'b', 'c', 'd'])
s2 = s1.reindex(['a', 'c', 'g'])
s2
# Différents types d'index
# et la conséquence...
s1 = pd.Series([1, 2, 3], index=[0, 1, 2])

# Les index sous forme de chaines
s2 = pd.Series([4, 5, 6], index=['0', '1', '2'])
s1 + s2
# réindexation en castant le type de l'index
s2.index = s2.index.values.astype(int)
s1 + s2
# On remplace par 0
s3 = s2.copy()
s3.reindex([0, 5], fill_value=0)
# Génération d'une série aléatoire d e3 lignes
np.random.seed(123456)
s = pd.Series(np.random.randn(3), index=['a', 'b', 'c'])
s
# On va créer une nouvelle ligne
s['d'] = 100
s
# on peut modifier cette valeur
s['d'] = -100
s
#  On peut supprimer un item en utilisant la fonction del
del(s['d'])
s
s = pd.Series(np.arange(50,60), index=np.arange(10,20))
s
#Prenons les items de la position 1 à la position 7 avec un delta de 2
s[1:7:2]
#On peut aussi simuler la même résultat qua la fonction head
s[:5]
# Idem pour la fonciotn tail
s[-5:]
# Si on veut tout à partir d'une position donnée
s[6:]
# si on veut tout à partir de l'indice 3 avec un pas de 2
s[3::2]
# Une possibilité très utile pour inverser la série
s[::-1]
# En partant du 5° enregistrement depuis la fin avec un pas de 2
s[5::-2]
# Tout sauf les 3 derniers
s[:-3]
# Que les 3 derniers
s[-3:]
# On peut combiner les deux pour ne garder que les 3 premiers des 4 derniers
s[-4:-1]
copie = s.copy()
separation = copie[:5]
separation[10]=666
copie[10]

s
s = pd.Series(np.arange(0, 6),
index=['a', 'b', 'c', 'd', 'e', 'f'])
s[0:3]
s['b':'e']