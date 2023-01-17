print('Bonjour tout le monde !')
2+2
3.14159 * 3**2
x = 2 
x
y = x+1

x += 1

print(x, y)
a = 10

b = 5

print(a == a)

print(a != b)

print(a < b)

print(a <= b)

print(a > b)

print(a >= b)
# Table de multiplication par 9

i = 0

while i<10 :

    print(i , " x 9 = ", i*9)

    i = i+1
for i in range(10) :

    print(i , " x 9 = ", i*9)
for i in range(100,10,-5) :

    print(i)
a = 10

b = 5

if a > b :

    print("le max est : ", a)

else :

    print("le max est : ", b)
def maximum(a,b) :

    """Affiche le max de deux éléments"""

    if a > b :

        print("le max est : ", a)

    else :

        print("le max est : ", b)
maximum(1,100)
help(maximum)
def maximum(a,b) :

    """Renvoie le max de deux éléments"""

    if a > b :

        max = a

    else :

        max = b

    return max
print(10*maximum(1,100))
chiffres = [0,1,2,3,4,5,6,7,8,9]
chiffres[3]
chiffres[-1]
len(chiffres)
voyelles = ['a','e','i','o','u','y']

liste = chiffres + voyelles

print(liste)
chiffres[3:5]
chiffres[:5]
chiffres[5:]
liste1 = ['p','y','t','h','o','n']
liste2 = liste1
liste1[3] = 1000
print(liste1)

print(liste2)
liste1 = ['p','y','t','h','o','n']
liste2 = list(liste1)
liste1[3] = 1000

print(liste1)

print(liste2)
for x in liste :

    print(x)
envers = []

for x in liste :

    envers = [x] + envers

print(envers)
envers = []

i=0

while i < len(liste) :

    envers = [liste[i]] + envers

    i += 1

print(envers)
[2*i for i in range(30,15,-2)]
def inverse(liste):

    return [liste[index] for index in range(len(liste)-1, 0, -1)]
inverse(chiffres)
def echange(liste, firstIndex, secondIndex):

    liste[firstIndex], liste[secondIndex] = liste[secondIndex], liste[firstIndex]

    return liste
echange(chiffres, 2, 8)
chiffres
def tri_a_bulles(liste):

    for i in range(len(liste)):

        for j in range(0, len(liste)-i-1):

            if liste[j+1] < liste[j]:

                liste = echange(liste, j, j+1)

    return liste
tri_a_bulles(inverse(voyelles))
inverse(voyelles)
nom = 'Le Cun'

prenom = 'Yann'
print(prenom,nom)
prenom_nom = prenom + ' ' + nom
prenom_nom
nom[0]
for c in nom : 

    print(c)
list(nom)
def palindrome(mot):

    liste = list(mot)

    for i in range(len(liste)):

            if (liste[i] != liste[-i-1]):

                return False

    return True
palindrome('python')
palindrome('radar')
import numpy as np
a = np.array(chiffres)

print(a)
a = np.arange(0,16)
a
b = a.reshape(4,4)

print(b)
b[3,2]
b[2:4, 1:3]
b.reshape(-1)