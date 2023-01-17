print("Max, Mustermann, 01.01.2020, München")
print ("Geben Sie ihren Vornamen")
Vorname=input("Vorname: ")
print ("Geben Sie ihren Nachnamen ")
Nachname=input("Nachname: ")
print ("Geben Sie ihren Geburtsort: ")
Geburtsort=input("Geburtsort: ")

print(Vorname,",",Nachname,",",Geburtsort)
print("Geben Sie die Größen für die Seiten a und b und c an")
a=int(input("a: "))
b=int(input("b: "))
c=int(input("c: "))
Volumen=a*b*c
Oberfläche=2*(a*c+b*c+a*b)
print("Volumen =",Volumen)
print("Oberfläche =",Oberfläche)
preis=float(input("Wenn du den Nettopreis ausrechnen willst gebe eine 1 und für den Bruttopreis eine 2: "))
Wert=float(input("Bitte gebe den Wert des Produktes an: "))
if preis == 1:
    Nettopreis=Wert*100/119
    print ("Nettopreis lautet: {0:.2f}".format(Nettopreis))
   
else:
    Bruttopreis=Wert*1.19
    print("Bruttopreis lautet: {0:.2f}".format(Bruttopreis))
   
Rabatt=float(input("Bitte geben sie an wie viel Rabatt sie bekommen: "))
Kaufpreis=float(input("Kaufpreis des Produktes: "))

if Rabatt>70:
    print("So viel Rabatt geht nicht")
else:
    Rabattpreis=Kaufpreis*Rabatt/100
    Endpreis=Kaufpreis-Rabattpreis
    print("Rabatt = {0:.2f}".format(Rabattpreis))
    print("Endpreis = {0:.2f}".format(Endpreis))
    
Kilo=float(input("Bitte geben sie ihr Gewicht in kg an: "))
Größe=float(input("Bitte geben sie ihre Größe in m an "))
BMI=Kilo/(Größe*Größe)

if BMI>25:
    print("Sie sind übergewichtig")
    
else:
    print("Sie sind nicht übergewichtig")
    
Kaufpreis=float(input("Kunde kauft in Wert von: "))
Kundegeld=float(input("Kunde gibt: "))

if Kundegeld<Kaufpreis:
    print("Kaufvorgang wird storniert")
elif Kundegeld>Kaufpreis:
    Rückgeld=Kundegeld-Kaufpreis
    print("Kunde bekommt: {0:.2f}".format(Rückgeld))
else:
    print("Kunde bekommt nichts zurück hat passend gezahlt")

L=[5,7,8,2,3,4]
print("Liste L =",L)
L.append(1)
L.append(6)
print("Neue Liste L =",L)
L.sort()
print("Sortierte Liste L =",L)
Text=input("Schreiben sie einen Text: ")
print("Der Text hat eine Buchstabenanzahl von: ",len(Text))
Text.upper
print("Der Text wird Groß und 3 mal angezeigt: ",3*Text)
a=int(input("Bitte geben Sie eine Zahl für a: "))
b=int(input("Bitte geben Sie eine Zahl für b: "))
print("a ist größer als b: ",bool(a>b))
if a==b:
    print ("beide Zahlen gleich groß")
Liste=["z","c","f","g","h","i","j","l"]
Liste.append("a")
Liste.append("b")
print("Liste =",Liste)
Liste[0]="e"
Liste[7]="d"
Liste.sort()
print("Sortierte Liste =",Liste)
print("Jedes 2te Element =",Liste[::2])
a = int(input("Bitte bestimme eine Zahl zwischen 18 und 50 a="))
if a < 20:
    print("Zahl ist unter 20 ")
elif 20 < a < 30:
    print("Zahl ist unter 30 über 20")
elif 30 < a < 40:
    print("Zahl ist unter 40 über 30")
else:
    print("Zahl ist unter 50 über 40")
s1={1,9,2,8}
s2={3,7,4,6}
s3={1,5,9,10}
s4={3,8,9,10}
print("Alle Sets ausgeben =",s1|s2|s3|s4)
print("Elemente die beide Sets haben =",s3&s4)
print("Elemente die s1 hat =",s1)
print("Elemente die beide Sets haben =",s1&s3)
print("Elemente die s3 hat aber nicht s4=", s3-s4)

x = 30 
if x > 40:
    print(x, "ist größer als 40") 
elif x == 40:
    print(x, "ist 40")
elif x > 5:
    print(x, "ist größer als 5")
else:
    print(x, "ist nicht definierbar")
##Erste Aufgabe
for x in range (1,10):
    print (x)
##Zweite Aufgabe
for x in [1, 3, 8, 9]:
    print(x, end= ' ')
##Dritte Aufgabe
x = 2
while x < 5:
    print(x)
    x = x + 1
for durchgang in range(10):
    if durchgang == 5:
        print("Abbruch")
        break
    print(durchgang)

for durchgang in range(10):
    if durchgang == 5: 
        print("Pause eingelegt")
        continue
    print(durchgang)
add = lambda a, b, c: a + b + c
add(2, 7, 1)
sorted([8, 3, 4, 1, 9, 7, 10, 2, 5])
##Schuhliste
data = [{'Marke': 'Nike', 'Preis': '50', 'Größe':39},
        {'Marke': 'Adidas', 'Preis': '60',     'Größe':42},
        {'Marke': 'Puma' , 'Preis': '30',     'Größe':43}]
##Schuhliste nach Alphabet der Marke
data = [{'Marke': 'Nike', 'Preis': '50', 'Größe':39},
        {'Marke': 'Adidas', 'Preis': '60',     'Größe':42},
        {'Marke': 'Puma' , 'Preis': '30',     'Größe':43}]

# sortierung
sorted (data, key=lambda item: item['Marke'])
## Sortierung nach der Schuhgröße

data = [{'Marke': 'Nike', 'Preis': '50', 'Größe':39},
        {'Marke': 'Adidas', 'Preis': '60',     'Größe':42},
        {'Marke': 'Puma' , 'Preis': '30',     'Größe':43}]

##Sortierung
sorted(data, key=lambda item: item['Größe'])
x = 50

if x > 0:
  raise Exception("x darf nicht positiv sein!")
string22 = "PyHton iSt cOoL"
string22.lower()
string22.upper()
string25 = "manfred programmiert gerne mit python."

string25.title()
string25.capitalize()
Zeile27 = '       Python      '
Zeile27.strip()
Zeile27.lstrip()
Zeile27.rstrip()
Zahlen28 = "111111840361111"
Zahlen28.strip('1')
Text29 = "Landshut"
Text29.center(16)
Text29.rjust(20)
Text29.ljust(20)
text30 = "Landshut ist eine schöne Stadt"
text30.find('Stadt')
text30.replace('schöne', 'große')
text31 = "Willkommen in der Stadt Landshut"
text31.split()
text32 = "Willkommen in, der Stadt, Landshut"
text32.split(",")
text33 = "Willkommen#in#der#Stadt#Landshut"
text33.split("#")
[i for i in range (30) if i % 4 > 0]

[n ** 2 for n in range (9)]
[(i,j) for i in range (4) for j in range (4)]
L = []
for var in range (36):
    if var % 2:
        L.append(var)
        
L
[val if val % 3 else -val
 for val in range(15) if val % 5]

def primzahlen(P):
    primzahlen = set()
    for i in range (2,P):
        if all (i % p > 0 for p in primzahlen):
            primzahlen.add(i)
            yield i
        
print(*primzahlen(38))
from math import *
cos(pi) ** 2 + tan(pi) ** 2