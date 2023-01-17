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
print(after)
# convert it back to utf-8--actually she is saying string
print(after.decode("ascii"))

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(
# Your turn! Try encoding and decoding different symbols to ASCII and
# see what happens. I'd recommend $, #, 你好 and नमस्ते but feel free to
# try other characters. What happens? When would this cause problems?
lang= 'türkçe%#$Turkish'
print('{} is my string'.format(lang) ,'\n')
utf = lang.encode('utf-8')
print('{} is the encoded format of my string in utf-8'.format(utf),'\n')
decodedutf = utf.decode('utf-8')
print('{} is the decoded format of my string from utf-8 '.format(decodedutf),'\n')
asci = lang.encode('ascii',errors='replace')
print('{} is the encoded format of my string in ascii'.format(asci),'\n')
decodedasci = asci.decode('ascii')
print('{} is the decoded format of my string from ascii'.format(decodedasci),'\n')
print('As we can see above it threw an error --question marks are not supposed be there-- in the case of ASCII') 
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
policek= "../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv"
with open(policek) as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)
print('I wanted to see what rb for')
# look at the first ten thousand bytes to guess the character encoding

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",'rb') as rawdatam:
    resultm = chardet.detect(rawdatam.read(1000))

# check what the character encoding might be
print(resultm)
print('Wow confidence is 100%')
# look at the first ten thousand bytes to guess the character encoding
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000000))
print(result)


police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='ascii')
police_killings.head()
police_killings_v = pd.read_csv(policek,encoding='Windows-1252')
police_killings_v.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv('police-shooting-data-utf-8.csv')
