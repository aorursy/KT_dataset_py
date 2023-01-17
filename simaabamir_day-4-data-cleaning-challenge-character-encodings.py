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

# start with a string
# before1 = "This is the dollar symbol: $"
# before1 = "This is the hash symbol: #"
before1 = "This is the namaste symbol: नमस्ते"

# check to see what datatype it is
print(type(before1))

# encode it to a different encoding, replacing characters that raise errors
after1 = before1.encode("utf-8", errors = "replace")

# check the type
print(type(after1))
# take a look at what the bytes look like
print(after1)

# convert it back to utf-8
print(after1.decode("utf-8"))
# try to decode our bytes with the ascii encoding
# print(after1.decode("ascii")) #this generates error

# start with a string
# before2 = "This is the dollar symbol: $" 
# before2 = "This is the hash symbol: #"
before2 = "This is the chinese text: 你好"

# encode it to a different encoding, replacing characters that raise errors
after2 = before2.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after2.decode("ascii"))

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(
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
    result2 = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result2)
# Interesting to see here that even confidence is 100%, yet it's not 'ascii'. Need to increase read size.
# read in the file with the encoding detected by chardet
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='ascii')

# look at the first few lines of police_killings
police_killings.head()
# look at the first hundred thousand bytes to guess the character encoding
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result2 = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result2)
# Interesting to see here that even confidence is 100%, yet it's not 'ascii'. Need to increase read size.
# read in the file with the encoding detected by chardet
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')

# look at the first few lines of police_killings
police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("ks-projects-201801-utf8.csv")
#function to detect encoding- universal detector.
import urllib
from chardet.universaldetector import UniversalDetector

usock = open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb')
detector = UniversalDetector()
for line in usock.readlines():
    detector.feed(line)
    if detector.done: break
detector.close()
usock.close()
print (detector.result)
import urllib
from chardet.universaldetector import UniversalDetector

usock = open("../input/die_ISO-8859-1.txt", 'rb')
detector = UniversalDetector()
for line in usock.readlines():
    detector.feed(line)
    if detector.done: break
detector.close()
usock.close()
print (detector.result)
