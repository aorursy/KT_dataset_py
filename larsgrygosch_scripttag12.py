print("Hallo Welt")
print(2)
print(7/2)
print(3 < 42)
print("Hallo", "Welt")
variablenname = "Wert"
print(variablenname)
print(id(variablenname))
print(hex(id(variablenname)))
var1 = 2
var2 = 42
print(hex(id(var1)))
print(hex(id(var2)))
var1 = 2
var2 = 2
var3 = 2.0
print(hex(id(var1)))
print(hex(id(var2)))
print(hex(id(var3)))
print(type(var1))
print(type(var3))
antwort = 42
pi = 3.14
text = "'Hallo Welt'"
wahrheit = False
leer = None
print(antwort,"gehört zum Datentyp:", type(antwort))
print(pi,"gehört zum Datentyp:", type(pi))
print(text,"gehört zum Datentyp:", type(text))
print(wahrheit,"gehört zum Datentyp:", type(wahrheit))
print(leer,"gehört zum Datentyp:", type(leer))
print(2 + 14) #Addition
print(2 - 5)  #Subtraktion
print(2 * 3)  #Multiplikation
print(12/7)  #Division
print(12//7) #Division ohne Rest (Wie häufig passt der Divisor in den Dividend)
print(12%7)  #Rest (Modulo)
import sys

print(sys.float_info)
print(sys.int_info)
var1 = 2.1**1025
print(var1)
var1 = 9/7**350
print(var1)
var1 = 9/7**400
print(var1)
print(1.1 + 2.2)
var2 = 1.000000000000001
var3 = 1.0000000000000001

print(var2 > 1)
print(var3 > 1)
print("1 == 2 ist: ", 1 == 2)   #genau gleich
print("1 <= 2 ist: ", 1 <= 2)   #kleiner-gleich
print("1 < 2 ist: ", 1 < 2)     #kleiner als
print("1 >= 2 ist: ", 1 >= 2)   #größer-gleich
print("1 > 2 ist: ", 1 > 2)     #größer als
print("1 != 2 ist: ", 1 != 2)   #nicht gleich
var1 = 33.5
print(var1)
var1 = int(var1)
print(var1)
var1 = bool(var1)
print(var1)
var1 = float(var1)
print(var1)
blindtext = "Lorem ipsum dolor sit amet, consectetuer adipiscing elit."
print(blindtext)
ref = None
print(ref)
print(None == 0)
a = 5 + 5
b = 5 + 5.0
c = 5 // 2
d = 5.0 // 2
e = "17" + "12"
f = 5 < 6

zahl = 5

if zahl > 0:
    print("Die Zahl ist positiv!")
zahl = 5

if zahl > 0:
    print("Die Zahl ist positiv!")
    
print("...folgender Programmteil")
zahl = -99

if zahl > 0:
    print("Die Zahl ist positiv!")
    
print("...folgender Programmteil")
zahl = -99

if zahl > 0:
    print("Die Zahl ist positiv")
elif zahl < 0:
    print("Die Zahl ist negativ")
else:
    print("Die Zahl ist 0")
zahl = 0

while zahl < 5:
    print(zahl, "ist kleiner als 5")
    zahl = zahl + 1
zahl = 0

while True:
    print(zahl, "ist kleiner als 5")
    zahl += 1
    if zahl == 5:
        break
for i in range(5):
    print("i =",i)
for i in range(2, 8):
    print("i =",i)


x = 3

for y in range(-3, 3):
    print(x/y)
x = 3

for y in range(-3, 3):
    try:
        print(x/y)
    except ZeroDivisionError:
        print("Teilen durch null!!")
x = 3

for y in range(-3, 3):
    try:
        print(x/y)
    except ZeroDivisionError:
        print("Teilen durch null!!")
        del x
    except NameError:
        print("x ist nicht definiert...")
test_text = "Das ist ein Test"
print(test_text)
print("a" in test_text)
print("Test" in test_text)
print(test_text[4])
print(test_text[2:10:2])
text = "Lorem ipsum dolor sit amet, consectetuer adipiscing elit."
print(text.lower())
print(text.upper())
print(text.startswith("Lorem"))
print(text.endswith("elit"))
print(text.index("o"))



