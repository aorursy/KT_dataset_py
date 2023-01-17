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

symbols_before = "$, #, 你好, नमस्ते"

# encode it to a different encoding, replacing characters that raise errors
symbols_after = symbols_before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(symbols_after.decode("ascii"))
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
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")
# look at the first ten thousand bytes to guess the character encoding
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result)
# read in the file with the encoding detected by chardet
PoliceKillingsUS = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')

# look at the first few lines
PoliceKillingsUS.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
PoliceKillingsUS.to_csv("PoliceKillingsUS-utf8.csv")
file_guide = pd.read_csv("../input/character-encoding-examples/file_guide.csv")
file_guide.head()
with open("../input/character-encoding-examples/olaf_Windows-1251.txt", 'r', encoding="Windows-1251") as rawdata:
    olaf = rawdata.read(5000)

# check what the character encoding might be
print(olaf)
with open("../input/character-encoding-examples/die_ISO-8859-1.txt", 'r', encoding="ISO-8859-1") as rawdata:
    die = rawdata.read(5000)

# check what the character encoding might be
print(die)
with open("../input/character-encoding-examples/harpers_ASCII.txt", 'r', encoding="ascii") as rawdata:
    harpers = rawdata.read(5000)

# check what the character encoding might be
print(harpers)
with open("../input/character-encoding-examples/portugal_ISO-8859-1.txt", 'r', encoding="ISO-8859-1") as rawdata:
    portugal = rawdata.read(5000)

# check what the character encoding might be
print(portugal)
with open("../input/character-encoding-examples/shisei_UTF-8.txt", 'r', encoding="utf-8") as rawdata:
    shisei = rawdata.read(1500)

# check what the character encoding might be
print(shisei)
with open("../input/character-encoding-examples/yan_BIG-5.txt", 'r', encoding="Big5") as rawdata:
    yan = rawdata.read(5000)

# check what the character encoding might be
print(yan)
import csv
with open("yan-utf8.csv",'w') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(yan)
    
with open("portugal-utf8.csv",'w') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(portugal)
    
with open("shisei-utf8.csv",'w') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(shisei)
    
with open("olaf-utf8.csv",'w') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(olaf)
    
with open("harpers-utf8.csv",'w') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(harpers)
    
with open("die-utf8.csv",'w') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(die)