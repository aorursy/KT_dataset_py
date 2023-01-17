# QUEL EST LE NOM QUE VOUS UTILISEZ POUR VOUS IDENTIFIER

NAME = "MULLER"

# AVEC QUI AVEZ VOUS TRAVAILLE

COLLABORATEURS = " "
print (f_4_2(0))
def f_4_2 (x):

    """ 

    La fonction f_4_2 est une fonction de coefficient directeur '4' 

    et d'ordonnée à l'origine '2'

    """

    # Ecrivez le resultat renvoyé (return) par 

    # La fonction f=4x+2

    # YOUR CODE HERE

    x=float(x)

    

    y= x*4+2

    

    return y
print(f_4_2(0))
""" Tests de validation automatique """

assert f_4_2(0) == 2

assert f_4_2(1)-2 == 4

# Testez ici votre fonction
def f_moins5_dix (x):

    """ 

    La fonction f_moins5_dix prends des valeurs de type float 

    et renvoie également des nombre flotants

    """

    #Ecrivez le resultat renvoyé (return) par

    # La fonction f(x)=-5x+10

    # YOUR CODE HERE

    x=float(x)

    

    y= -5*x+10

    

    return y

print (f_moins5_dix (0))
""" Verifier que la fonction renvoie bien des nombres réels """

assert f_moins5_dix('3') == -5.0

assert f_moins5_dix('3.3') == -6.5

assert f_moins5_dix('3.33') == -6.649999999999999

def f_moins1_moins3 (x):

 # YOUR CODE HERE

 x=float(x)

 y=(-1)*x-3   

 return y

# Testez ici votre fonction :

print(f_moins1_moins3(1))

""" Verifier que la fonction renvoie bien des nombres réels """

assert f_moins1_moins3 (10) == -13

assert f_moins1_moins3 (0) == -3

def pente (fonction_affine):

    """

    Cette fonction a pour variable une fonction affine

    elle doit retrouver le coefficient directeur 'a' d'une

    telle fonction 

    """

    # Ecrivez le resultat renvoyé (return) 

    # YOUR CODE HERE

    (fonction_affine)=float(fonction_affine)

    a=(fonction_affine)-b/(x-0)

    return a
# Testez ici votre fonction :

x=1

b=2

pente(6)
assert pente (f_4_2) == 4
def changement_signe (fonction_affine):

    """

    Cette fonction a pour variable une fonction affine

    elle doit retrouver l'abscisse x où il y a changement du signe

    d'une telle fonction 

    """

    # Ecrivez le resultat renvoyé (return) 

    # YOUR CODE HERE

    (fonction_affine)=float(fonction_affine)

    x=(-b)+(fonction_affine)/a

    return(x)
# Testez ici votre fonction

b=2

a=4

changement_signe (0)
""" Tests de validation automatique """

assert changement_signe (f_4_2) == -0.5
# Entréz votre code ci dessous pour définir une fonction comme dans les cas précédents : 



# YOUR CODE HERE

raise NotImplementedError()
# Testez ici votre fonction
""" Tests de validation automatique """

for x in [0,2,4,5,50] :

    assert racine(x) == pow(x,0.5)

    
def BC (AB,AC):

    """

    Calcul de la longueur de l'hypothénuse 

    en utilisant le théorème de pythagore 

    et la fonction racine définie préalablement

    """

    # YOUR CODE HERE

    raise NotImplementedError()
# Testez ici votre fonction
""" Tests de validation automatique """

assert BC(1,1) == pow(2,0.5)

assert BC(2,1) == pow(5,0.5)
