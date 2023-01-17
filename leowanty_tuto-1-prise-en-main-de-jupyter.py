# Les éléments en commentaire n'affectent pas l'execution du code

# et n'affichent rien en sortie de cellule

print("Hello world! Et si on calculait 1+1 ?")

1+1
# La réponse s'affiche, c'est top !

# Maintenant, on enregistre ça dans une variable *resultat*

print("Hello world! Et si on calculait 1+1 ?")

resultat = 1+1
# Comme on a enregistré le résultat dans une variable, il ne s'affiche pas.

print("Hello world! Et si on calculait 1+1 ?")

print(resultat)



# On en profite pour montrer que si le calcul n'est pas présenté en dernier, alors il n'est pas affiché :

print("Hello world! Et si on calculait 2+2 ?")

2+2

print("Ah zut, si je n'étais pas si bavard, on aurait pu connaître le résultat...")
# Je nomme deux listes d'éléments, on prendra soin de la première et mal de la seconde :

list_1 = ['Hello','World!','Je','suis','triste!']

list_2 = ['Hello','World!','Je','suis','triste!']



print('liste 1 :', ' '.join(list_1))

print('liste 2 :', ' '.join(list_2))
# Je fais mes manipulations pour rendre mes listes heureuses :

list_1[4] = 'heureux!'

list_2[4] = 'heureux!'



print('liste 1 :', ' '.join(list_1))

print('liste 2 :', ' '.join(list_2))
# Ici, je ne fais pas de manipulation de mes listes, juste l'affichage :

# (Par contre, regardez bien le numéro du In[xx] de la cellule et des cellules avant et après...)

print('liste 1 :', ' '.join(list_1))

print('liste 2 :', ' '.join(list_2))



# Si les cellules ont été exécutées dans l'ordre, essayez d'appliquer la cellule ci-dessous AVANT d'exécuter cette cellule

# Vous verrez que l'ordre d'execution importe sur les résultats de la cellule.
# Ici, je manipule ma seconde liste :

list_2[4] = 'triste!'
# Python devine le type des valeurs qu'il enregistre en mémoire :

test1 = 15

test2 = 15.6

test3 = 'Hello World'



print('test 1 :')

print(type(test1))

print('\ntest2 :')

print(type(test2))

print('\ntest3 :')

print(type(test3))
# Les éléments simples peuvent prendre 

entier = int(15.6) # Quand une valeur décimale est forcée en entier (commande int), alors les décimales sont tronquées.

decimal = float(15) # Lorsqu'un entier est forcé en décimale (commande float), alors une décimale à 0 est ajoutée.

# Pour passer en décimal, on peut multiplier par un décimal : int(15) * 1.0   donnera le nombre décimal 15.0

boolean = bool(1) # Identique à boolean = True

chaine_de_caracteres = str('Hello World!')

manquant = None



print('entier :')

print(entier)

print('\ndecimal :')

print(decimal)

print('\nboolean :')

print(boolean)

print('\nchaine_de_caracteres :')

print(chaine_de_caracteres)

print('\nmanquant :')

print(manquant)

# Les collections d'éléments peuvent prendre des valeurs de types différets :

liste = [1,20,'3',40]

mon_tuple = (1,'100',10,1000)

ensemble = set([10,2,'a',2,5,0]) # L'ensemble que garde que les éléments distincts et ne garde pas l'ordre d'écriture des éléments

dictionnaire = {'cle1':1, 'cle3':'cle_numero_3', 'cle2':2, 'cle1':2} # Le dictionnaire inscrit chaque élément comme un couple clé:valeur .

# Il ne garde pas l'ordre d'écriture des éléments. Attention : si la même clé apparaît deux fois, la valeur sera écrasée par la dernière inscription.



print('liste :')

print(liste)

print('\nmon_tuple :')

print(mon_tuple)

print('\nensemble :')

print(ensemble)

print('\ndictionnaire :')

print(dictionnaire)
# on accède au second élément de chaque collection :

print('liste :')

print(liste[1]) # Le comptage des éléments par index commence à 0

print('\ntuple :')

print(mon_tuple[1])

print('\nensemble :')

print(list(ensemble)[1]) # Les ensembles set n'acceptent pas l'accès indexé à un elément. Il faut repasser par un autre type comme la liste.

print('\ndictionnaire :')

print(dictionnaire['cle2']) # Pour le dictionnaire, on accède à la valeur en nommant la clé ne garde pas l'ordre d'écriture des éléments



# Une chaine de caractères est aussi considérée comme une collection de lettres.

# En fait, on peut lui appliquer peu ou prou les mêmes traitement qu'une liste !

print('\nchaine de caractères :')

print(chaine_de_caracteres[1]) # On sélectionne le deuxième élément (la deuxième lettre) de la chaine de caractères
# Rretrouver trois éléménts de la collection :

print('La chaîne de caractères :')

print(chaine_de_caracteres)



print('\nLes trois premiers éléments de la chaîne de caractères :')

print(chaine_de_caracteres[:3])

print('\nLes trois derniers éléments de la chaîne de caractères :')

print(chaine_de_caracteres[-3:])

print('\nLes éléments 6 à 9 :')

print(chaine_de_caracteres[6:9])
test_numeric = 15 == 15.0

test_numeric2 = 18 < 7

test_character = 'llo' in 'Hello World' # Une sous-chaine de caractères peut être testée face à une chaine de caractères complète.



print('test_numeric :')

print(test_numeric)

print('\ntest_numeric2 :')

print(test_numeric2)

print('\ntest_character :') 

print(test_character)
print('Test classique :')

x = 5



if 3*x <12:

    print('x*3 est inférieur à 12')

elif 4*x >32 :

    print('x*4 est supérieur à 32')

else :

    print('x est compris entre 4 et 8')
print('VALEURS NUMERIQUES :')

# Valeur nulle :

x = None

print(x)

if x :

    print("x n'est pas vide\n")

else : print('x est vide\n')



# Valeur pleine :

x = 12

print(x)

if x :

    print("x n'est pas vide\n")

else : print('x est vide\n')



# X est égal à 0 :

x = 0

print(x)

if x :

    print("x n'est pas vide\n")

else : print('x est vide\n')
print('Chaînes de caractères :')

# Valeur pleine :

x = 'None'

print(x)

if x :

    print("x n'est pas vide\n")

else : print('x est vide\n')



# Chaîne vide :

x = ''

print(x)

if x :

    print("x n'est pas vide\n")

else : print('x est vide\n')
print('Ensembles :')

# Liste pleine :

x = [1,50,25,100]

print(x)

if x :

    print("x n'est pas vide\n")

else : print('x est vide\n')



# Liste vide :

x = []

print(x)

if x :

    print("x n'est pas vide\n")

else : print('x est vide\n')

    

# Dictionnaire vide :

x = {}

print(x)

if x :

    print("x n'est pas vide\n")

else : print('x est vide\n')
for x in range(3):

    print(x)
for element in [5,'Hello',12,'World!']:

    print(element)
ma_liste = [0,0,0,0,0]

for i in range(5) :

    ma_liste[i] = i*10

    

ma_liste2 = [i*10 for i in range(5)]



ma_liste3 = [i*10 for i in range(5) if i%2==0] # On affiche que si i est multiple de 2





print('Liste 1 :')

print(ma_liste)

print('\nListe 2 :')

print(ma_liste2)

print('\nListe 3 :')

print(ma_liste3)
print('Exemple avec continue :')

for element in [5,'Hello',12,'World!']:

    if type(element) == str :

        continue # Si la condition est remplie, on passe à l'élément suivant sans appliqué le code restant dans la boucle

    print(element)



print('\nExemple avec break :')

for element in [5,'Hello',12,'World!']:

    if type(element) == str :

        break # Si la condition est remplie, on arrête complètement la boucle 

    print(element) 
i=0

while i<5 :

    print(i)

    i = i+1
# Syntaxe de base, commence par def nom_de_fonction(arg1,arg2,...) :

# Et retourne le contenu de la commande return()

def fonction_addition(a,b) :

    return(a+b)



# La fonction s'arrête dès que le premier return est atteint 

def fonction_demo(a,b) :

    c = a

    return(c)

    c = a+b

    return(c)
print('fonction_addition :')

print(fonction_addition(5,10))

print('\nfonction_demo :')

print(fonction_demo(5,10))