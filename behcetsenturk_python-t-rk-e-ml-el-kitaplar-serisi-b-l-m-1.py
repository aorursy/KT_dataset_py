print("Hello, World!")
print("Hello, World!")
a = 5

print(a) # Integer



b = 5.5

print(b) # Float



c = "Hello!"

print(c) # String
"""Yorum 

Satırları

Docstring"""

print(a)
a = 5

A = 6

print(a,A)
print("a'nın tipi:", type(a))

print("b'nin tipi:", type(b))



a = str(a)

b = int(b)



print("a'nın tipi:", type(a))

print("b'nin tipi:", type(b))
a = 5

b = 6

print(a+b)
a = 5

b = 3



print("a + b =", a + b)

print("a - b =", a - b)

print("a * b =", a * b)

print("a / b =", a / b)

print("a % b =", a % b)

print("a ** b =", a ** b)
a = 5

b = 7



a += b # a = a + b

b **= 2 # b = b**2



print(a)

print(b)
a = 5

b = 5



print(a == b) # a b'ye eşittir.

print(a >= b) # a b'ye eşittir veya büyüktür.

print(a != b) # a b'den farklıdır.
a = 6

b = 6

c = 2

print(a == b or c == 5)

print(a == b and c == 3)
print("alp" in "alpha")

print("b" in "alpha")
liste = ["Alpha", "Gamma", "Delta"]

print(liste)
print(liste[0])
liste[0] = "Omega"

print(liste)
print(len(liste))
liste.append("Alpha")

print(liste)
liste.insert(2, "Epsilon")

print(liste)
liste.remove("Delta")

print(liste)
liste.sort()

print(liste)
print(liste)

print(liste[0:2])
print(liste[1:])
liste.append("Theta")

print(liste)

print(liste[1:5:2])
a = "Hello, World!"

print(type(a))

print(a[0])

print(len(a))
print(a.lower())
print(a.replace('World', 'Sekai'))
print(a[0:9])

print(a[2:11:3])
part1, part2 = a.split(',')

print(part1)

print(part2)
print(a.split('o'))

x, y, z = a.split('o')

print(x)

print(y)

print(z)
var = "Narita"

print("Burası "+var+" şehridir.")
tupleA = ("Alpha", "Bravo", "Charlie")

print(tupleA)
print(tupleA[1])
küme = {"Quadra", "Double", "Triple"}

print(küme)
küme.add("Hexa")

print(küme)
sözlük = {

    "Isim": "Otostopçunun Galaksi Rehberi",

    "Yazar": "Douglas Adams",

    "Dil": "Türkçe"

}

print(sözlük["Dil"])
sözlük["Dil"] = "Almanca"

print(sözlük)
sözlük["Tür"] = "Bilim Kurgu"

print(sözlük)
yağmur = 0



if (yağmur == 1):

    perde = 1

    lamba = "Yarım güç"

    print("Yağmur yağıyor")

    

else:

    print("Yağmur yağmıyor")
Elektrik = "Yok"

Benzin = "Var"



if Elektrik == "Yok" and Benzin == "Var":

    Jeneratörler = "Açık"

    print("Jeneratörler çalıştırılıyor.")
çalışma_modu = "Yarı Otomatik"



if çalışma_modu == "Tam Otomatik":

    print("Sistem Tam Otomatik Modda")

elif çalışma_modu == "Yarı Otomatik":

    print("Sistem Yarı Otomatik Modda")

else:

    print("Sistem Manuel Modda")
çalışma_modu = "Yarı Otomatik"



if çalışma_modu == "Tam Otomatik":

    print("Sistem Tam Otomatik Modda")

    if çalışma_modu == "Yarı Otomatik":

        print("Sistem Yarı Otomatik Modda")

else:

    print("Sistem Manuel Modda")
çalışma_modu = "Yarı Otomatik"



if "Otomatik" in çalışma_modu:

    if çalışma_modu == "Tam Otomatik":

        print("Sistem Tam Otomatik Modda")

    if çalışma_modu == "Yarı Otomatik":

        print("Sistem Yarı Otomatik Modda")

else:

    print("Sistem Manuel Modda")
sayaç = 0



while (sayaç < 3):

    print(sayaç)

    sayaç = sayaç + 1
sayaç = 0



while sayaç < 10:

    if sayaç % 2 == 0:

        print(sayaç)

    sayaç = sayaç + 1
print(list(range(10)))
print(range(10))

print(type(range(10)))
for i in range(3):

    print(i)
print(list(range(6,14,3)))
liste = list(range(4))



print(liste)



for i in liste:

    print(i)
liste = ["Uniform", "Delta", "Sierra", "Oscar"]



for i in liste:

    print("--> "+i)
print(liste)

print(len(liste))

print(range(len(liste)))

print("-------------------")



for i in range(len(liste)):

    print(i)
for index, i in enumerate(liste):

    print("index =", index, "/ iterator =", i)
for i in range(5):

    if i == 3:

        break

    print(i)

for i in range(5):

    if i == 3:

        continue

    print(i)

def kare_alan_fonksiyon(sayi):

    karesi = sayi * sayi

    return karesi

a = 7

sonuç = kare_alan_fonksiyon(a)



print(sonuç)
print(kare_alan_fonksiyon(11))
def bölme(sayiA, sayiB):

    

    if sayiB == 0:

        print("Sıfır ile bölünemez.")

        return 0, 0

    

    bölüm = int(sayiA / sayiB)

    kalan = sayiA % sayiB

    

    return bölüm, kalan
print(bölme(17, 5))
bölüm, kalan = bölme(21, 10)

print("Bölüm =", bölüm, "Kalan =", kalan)
bölüm, kalan = bölme(6, 0)

print("Bölüm =", bölüm, "Kalan =", kalan)
mod = lambda sayi1, sayi2 : sayi1 % sayi2

print(mod(11, 8))
liste = [1, 2, 3, 4, 5]

kareler = []

for i in liste:

    kareler.append(i**2)

    

print(kareler)
print(list(map(lambda x: x**2, list(range(1,6)))))
import math



print(math.sin(math.radians(30)))
!pip install colorama
liste = list(range(100))



istenen_liste = []



for i in liste:

    if i % 2 != 0:

        istenen_liste.append(i**2)



print(istenen_liste)
liste = list(range(100))



print([i**2 for i in liste if i % 2 != 0])



# Tek Satırda = print([i**2 for i in list(range(100)) if i % 2 != 0])
import os



os.listdir("../input/") 
os.mkdir("./test")

os.listdir(".")
os.rmdir("./test")

os.listdir(".")