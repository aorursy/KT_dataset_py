name = "murat"
3*name
"murat" + "sahin"
"m" + name[1:]
names = ["murat", "ali", "veli"]

for i in names:

    print("Name", i, sep=" : ")
print(*enumerate(names))
for i in enumerate(names):

    print(i)
for i in enumerate(names, 5):

    print(i)
for i in enumerate("names"):

    print(i)
"murat".isalpha()
"murat30".isalpha()
"123".isnumeric()
"123".isdigit()
"murat30".isalnum()
name = "muratsahin"
name[0:2]
name.index("a")
name.index("a",4)
name.startswith("m")
name.endswith("n")
name.count("a")
sorted("defter")
print(*sorted("defter"), sep = " ")
name = "murat sahin"

name.split()
name.split("a")
name.upper()
name.lower()
name_b = name.upper()

name_b
name_b.islower()
name_b.isupper()
name.capitalize()
name = name.title()

name
name.swapcase()
name = " murat "

name
name.strip()
name = "99hello99"

name.strip("9")
name.lstrip("9")
name.rstrip("9")
name = "murat sahin"

splitted = name.split()

splitted
joiner = " || "

joiner.join(splitted)
name
name.replace("a","x")
expression = " Ç ş Ş ç İ Ö ö ğ Ğ Ü ü "

lettersOld = "çÇşŞıİöÖüÜğĞ"

lettersNew = "cCsSiIoOuUgG"
alphabets = str.maketrans(lettersOld, lettersNew)

expression.translate(alphabets)
import pandas as pd

names = ["ayse", "Ayse", "ali", "Ali", "aali","veli", "mehmet"]

seri = pd.Series(names)

seri
seri.str.contains("al")
seri[seri.str.contains("al")]
seri.str.contains("al").sum()
seri.str.contains("[aA]li")
seri.str.contains("[aA]li").sum()