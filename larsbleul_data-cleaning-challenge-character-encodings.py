# modules we'll use

import pandas as pd

import numpy as np



# helpful character encoding module

import chardet



# set seed for reproducibility

np.random.seed(0)
# start with a string

before = "This is the euro symbol: €"



# check to see what datatype it is

type(before)
# encode it to a different encoding, replacing characters that raise errors

after = before.encode("utf-8", errors = "replace")



# check the type

type(after)
# take a look at what the bytes look like

after
# convert it back to utf-8

print(after.decode("utf-8"))
# try to decode our bytes with the ascii encoding

print(after.decode("ascii"))
# start with a string

before = "This is the euro symbol: €"



# encode it to a different encoding, replacing characters that raise errors

after = before.encode("ascii", errors = "replace")



# convert it back to utf-8

print(after.decode("ascii"))



# We've lost the original underlying byte string! It's been 

# replaced with the underlying byte string for the unknown character :(
# Your turn! Try encoding and decoding different symbols to ASCII and

# see what happens. I'd recommend $, #, 你好 and नमस्ते but feel free to

# try other characters. What happens? When would this cause problems?



before_dollar = "$"

before_hashtag = "#"

before_china = "你好"

before_whatever = "नमस्ते"



after_dollar = before_dollar.encode("ascii", errors = "replace")

after_hashtag = before_hashtag.encode("ascii", errors = "replace")

after_china = before_china.encode("ascii", errors = "replace")

after_whatever = before_whatever.encode("ascii", errors = "replace")



print(after_dollar.decode("ascii"))

print(after_hashtag.decode("ascii"))

print(after_china.decode("ascii"))

print(after_whatever.decode("ascii"))



# Dollar und Hashtag werden ohne Probleme ver- und entschlüsselt. Bei den beiden anderen Sonderzeichen ging der originale String verloren.

# Dies kann problematisch werden bei länderspezifischen Daten und wenn man Daten ersetzen will.
# try to read in a file not in UTF-8

kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")
# look at the first ten thousand bytes to guess the character encoding

with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:

    result = chardet.detect(rawdata.read(10000))



# check what the character encoding might be

print(result)
# read in the file with the encoding detected by chardet

kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252')



# look at the first few lines

kickstarter_2016.head()
# Your Turn! Trying to read in this file gives you an error. Figure out

# what the correct encoding should be and read in the file. :)



# police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")





# ERSTE PRÜFUNG:



#with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:

    #result = chardet.detect(rawdata.read(10000))

#print(result)



# Ergebnis: 100% Wahrscheinlichkeit das es sich um ascii handelt aber das ist falsch. Die Anzeige würde wieder einen Fehler produzieren.





# ZWEITE PRÜFUNG:

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:

    result = chardet.detect(rawdata.read(10000000000000))

print(result)



# Ergebnis: 73% Wahrscheinlichkeit das es sich um Windows-1252 handelt.



police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')

police_killings.head()

# Der Fehler ist behoben
# save our file (will be saved as UTF-8 by default!)

kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding

police_killings.to_csv("police-killings-utf8.csv")