import numpy as np

import pandas as pd
# Transformation d'une liste en vecteur :

ma_liste = [1,20,13,40,25]

print('ma_liste :')

print(ma_liste)

mon_vecteur = np.array(ma_liste)

print('\nmon vecteur :')

print(mon_vecteur)



# Taille d'un vecteur :

print('\nTaille de mon vecteur (même appel que pour une liste) :')

print(len(mon_vecteur))

print('\nDimension de mon vecteur (même appel que pour une matrice) :')

print(mon_vecteur.shape) # Retourne un tupple
# Création d'un vecteur de taille n :

n=5

mon_vecteur = np.repeat(10, n)

print('Vecteur de répétant une valeur :')

print(mon_vecteur)



mon_vecteur = np.zeros(n)

print('\nVecteur de 0 :')

print(mon_vecteur)



mon_vecteur = np.ones(n)

print('\nVecteur de 1 :')

print(mon_vecteur)



mon_vecteur = np.linspace(0,10,n)

print('\nVecteur de n valeurs a intervalles réguliers entre 0 et 10 :')

print(mon_vecteur)
mon_vecteur = np.linspace(0,10,5)



print('Multiplication par un scalaire :')

print(mon_vecteur * 2)



print('\nComparaison à un scalaire :')

print(mon_vecteur > 5)



print('\nOpération avec un autre vecteur :')

print(mon_vecteur + np.ones(5))



print('\nComparaison avec un autre vecteur :')

print(mon_vecteur > np.repeat(4,5))



print('\nProduit vectoriel :')

print(np.dot(mon_vecteur, np.repeat(2,5)))



def custom_function(x):

    if x%5==0 : # Si x est un multiple de 5 (i.e. x modulo 5 = 0)

        y = x**2 # alors y = x²

    else : y = x # sinon y = x

    return(y)

vectorized_function = np.vectorize(custom_function)

print('\nAppliquer une fonction construite à la main, élément par élément :')

print(vectorized_function(mon_vecteur))

# Avec une boucle on pourrait faire : np.array([custom_function(x) for x in mon_vecteur])
mon_vecteur = np.linspace(0,100,11) # Vecteur de 11 éléménts



print('Mon vecteur :')

print(mon_vecteur)



print('\n5ème élément du vecteur :') # Attention : l'indice du premier élément d'un vecteur est toujours 0

print(mon_vecteur[4])



print('\n3ème élément en partant de la fin :') # Le premier élément en partant de la fin est obtenu par l'index -1

print(mon_vecteur[-3])



print('\n2ème au 5ème éléments : ')

print(mon_vecteur[1:5])
# Mask pour ne récupérer que les lignes (index) paires :

mask_pair = [i%2==0 for i in range(len(mon_vecteur))]

print("Mask lignes (index) pair :")

print(mask_pair) # liste

print("Mon vecteur filtré :")

print(mon_vecteur[mask_pair])



# Mask pour ne récupérer que les éléments supérieurs à 50 :

mask_sup50 = mon_vecteur > 50

print("\nMask éléments supérieurs à 50 :")

print(mask_sup50) # np.array

print("Mon vecteur filtré :")

print(mon_vecteur[mask_sup50])
# Transformation d'une liste de listes en matrice :

ma_liste_de_listes = [

    [1,20,13,40,25],

    [5,8,75,3,12],

    [54,32,12,3,54],

    [5,78,5,66,4]

]

print('Ma liste de listes:')

print(ma_liste_de_listes)

ma_matrice = np.array(ma_liste_de_listes)

print('\nMa matrice :')

print(ma_matrice)



# Taille de la matrice :

print('\nDimension de mon vecteur :')

print(ma_matrice.shape) # Retourne un tuple

print('\nLa fonction de base len renvoie le nombre de lignes :')

print(len(ma_matrice))
# Créer une matrice avec les fonctions np.ones, np.zeros ...

print("Matrice de 1 :")

print(np.ones( shape=(4,5) ))



# Créer une matrice à partir d'une liste de valeurs :

liste_plate = [i for i in range(12)]

print("\nVecteur d'origine :")

print(liste_plate)

print("\nMatrice à 3 lignes et 4 colonnes :")

ma_matrice = np.reshape(liste_plate, (3,4))

print(ma_matrice)

print("\nOn réécrit la matrice en matrice à 6 lignes et 2 colonnes :")

print(ma_matrice.reshape(6, 2))
ma_matrice = np.reshape([i for i in range(20)], (5,4))

print("Ma matrice :")

print(ma_matrice)

print("Dimension de la matrice : "+str(ma_matrice.shape))



print("\nAccéder à l'élément en ligne 2 et colonne 3 :")

print(ma_matrice[1,2]) # Rappel, le premier index à toujours la valeur 0



print("\nOpération sur matrice (somme) :")

print(np.sum(ma_matrice))



print("\nOpération avec un scalaire :")

print(ma_matrice * 10)

print("\nPlusieurs opérations successives avec scalaires :")

print((ma_matrice-8)**2)



print("\nComparaison élément à élément, avec une autre matrice de dimension identique :")

print(ma_matrice < (ma_matrice-10)**2)



mask = ma_matrice < (ma_matrice-8)**2

ma_matrice[mask] = -1

print("\nRemplacer par -1 les valeurs de la matrice d'origine lorsque ses valeurs sont inférieures à celles de la matrice transformée :")

print(ma_matrice)
ma_matrice = np.reshape([i for i in range(20)], (5,4))

print("Ma matrice :")

print(ma_matrice)



mask_lignes = [True, False, True, True, False]

mask_colonnes = [False, True, True, False]

print("\nNe garder que les lignes 1, 3 et 4 :")

print(ma_matrice[mask_lignes,:])

print("Ne garder que les colonnes 2 et 3 :")

print(ma_matrice[:,mask_colonnes])



print("\nOpération par ligne (somme) :")

print(np.sum(ma_matrice, axis=0))

print("Opération par colonne (somme) :")

print(np.sum(ma_matrice, axis=1))



def custom_function(x):

    v_round = np.vectorize(round) # Vectorisation de la fonction d'arrondi

    y = v_round(x**2 / x.mean(),2)

    return(y)

print("\nFonction custom appliquée ligne par ligne :")

print(np.apply_along_axis(custom_function, axis=0, arr=ma_matrice))

print("Fonction custom appliquée colonne par colonne :")

print(np.apply_along_axis(custom_function, axis=1, arr=ma_matrice))



mon_vecteur = np.array([i**2 for i in range(9)])

ma_serie = pd.Series(mon_vecteur)

print("Mon vecteur :")

print(mon_vecteur)

print("Ma série :")

print(ma_serie)

print("Passer de la série pandas au vecteur numpy :")

print(ma_serie.values)



ma_matrice = mon_vecteur.reshape((3,3))

mon_df = pd.DataFrame(ma_matrice)

print("\nMa matrice :")

print(mon_vecteur)

print("Mon data frame :")

print(mon_df)

print("Passer du data frame pandas à la matrice numpy :")

print(mon_df.values)
mon_df.columns = ['X1','X2','X3']

mon_df.index = ["row1","row2","row3"]

print("Mon data frame après ajout de nom de colonnes et de lignes :")

print(mon_df)





print("\nOn récupère facilement les noms de colonnes et de lignes :")

print(mon_df.columns.values)

print(mon_df.index.values)
print('Afficher la colonne X1 sous forme de série :')

print(mon_df.X1)



print('-'*30)

print('\nAfficher la ligne row2 :')

print(mon_df.loc['row2'])

print('\nAfficher les colones X1 et X3 :')

print(mon_df.loc[:,['X1','X3']])

print('\nAfficher la colonne X2 des row1 et row3 :')

print(mon_df.loc[['row1','row3'],'X2'])



print('-'*30)

print('\nAfficher les deux premières lignes :')

print(mon_df.iloc[:2])

print('\nAfficher la colonne 2 :')

print(mon_df.iloc[:,2])

print('\nAfficher la colonne 1 et 3 de la dernière ligne :')

print(mon_df.iloc[-1,[0,2]])



print('-'*30)

print('\nAfficher la première ligne avec un mask :')

print(mon_df.iloc[[True,False,False]])

print('\nAfficher les colonnes 2 et 3 avec un mask :')

print(mon_df.iloc[:,[False,True,True]])

print('\nRemplacer les éléments diagonnaux par 0:')

v_bool = np.vectorize(bool) # Fonction vectorisée pour transformer chaque élément en booléen (les valeurs nulles/vides en False et le reste en True)

mask = v_bool(np.eye(mon_df.shape[0],mon_df.shape[1]))

mon_df[mask]=0

print(mon_df)
print("Ajouter une colonne 'Xnew' :")

mon_df['Xnew'] = [100,101,102]

print(mon_df)



print("Ajouter une ligne :")

mon_df.loc['rowNew'] = [1001,1002,1003,1004]

print(mon_df)
mon_df
mon_df.values
mon_df = pd.DataFrame([[21,3],[45,10],[0,24],[31,12],[7,3],[12,4],[9,5], [22,15],[8,7],[12,9],[10,36],[10,20],[1,3],[40,21]],

                      columns = ['X1','X2'],

                      index = ['client_'+str(i//4) for i in range(14)])

mon_df
mon_df['X_sum'] = mon_df.apply(np.sum,axis=1)

print("Création d'une colonne X_sum qui soit la somme par ligne :")

print(mon_df)



def substract_conditional(x): # x est un nombre scalaire entier

    if not isinstance(x,int) : # Si x n'est pas un entier, on renvoie la valeur -999

        return(-999)

    elif x<10 : # Sinon, si x est inférieur à 10, y = 0

        y = 0

    else : # Sinon, y = x

        y = x

    return(y)

print("\nMettre à 0 tous les éléments du DataFrame qui sont inférieurs à 10 :")

print(mon_df.applymap(substract_conditional))
print("Combine groupy-apply pour obtenir la variance par client (utilisation de l'index) :")

print(mon_df.groupby(level=0).apply(np.var))



print("\nCombine groupby-apply pour obtenir le max de deux lignes successives (utilisation d'un mask).\nOn affiche que la série X_sum :")

print(mon_df.groupby([i//2 for i in range(14)]).X_sum.apply(np.max))



def centrer_reduire(x): # Le x en entrée est un vecteur

    y = (x - x.mean()) / x.var()

    return(y) # Le y en sortie est un vecteur

print("\nCombine groupby-transform pour centrer-réduire les données (utilisation de l'index) :")

print(mon_df.groupby(level=0).transform(centrer_reduire))



print("\nTraitements plus complexe dans une boucle :")

for identifiant,valeurs in mon_df.groupby(level=0) :

    print(identifiant)

    print(valeurs)

    print('.. ..'*5)
mon_df = pd.DataFrame([[21,3],[45,10],[0,24],[31,12],[7,3],[12,4],[9,5], [22,15],[8,7],[12,9],[10,36],[10,20],[1,3],[40,21]],

                      columns = ['TS1','TS2'])

mon_df
print("Moyenne sur fenêtre glissante centrée de taille 5 :")

print(mon_df.rolling(5, center=True).apply(np.mean))