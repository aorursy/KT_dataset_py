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
# print(after.decode("ascii"))
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
symbol = " Hello नमस्ते "
after = symbol.encode("ascii", errors = "replace")
after.decode("ascii")
# try to read in a file not in UTF-8
# kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")
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
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    resultformat = chardet.detect(rawdata.read(100000)) #increased the number of sample to 100000

# check what the character encoding might be
print(resultformat)
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
police_killings.head()

# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("PoliceKillingsUS.csv")
with open("../input/character-encoding-examples/die_ISO-8859-1.txt", 'rb') as rawdata:
    resultformat = chardet.detect(rawdata.read(100000)) #increased the number of sample to 100000

# check what the character encoding might be
print(resultformat)
iso_df = pd.read_fwf("../input/character-encoding-examples/die_ISO-8859-1.txt", encoding='ISO-8859-1')
iso_df.head()
iso_df.to_csv("die_ISO-8859-1.csv")
with open("../input/die-iso88591csv/die_ISO-8859-1.csv", 'rb') as rawdata:
    resultformat = chardet.detect(rawdata.read(100000)) #increased the number of sample to 100000

# check what the character encoding might be
print(resultformat)
ascii_df = pd.read_fwf("../input/character-encoding-examples/harpers_ASCII.txt", encoding='ascii')
ascii_df.head()
ascii_df.to_csv("harpers_ASCII.csv")
windows_df = pd.read_fwf("../input/character-encoding-examples/olaf_Windows-1251.txt", encoding='Windows-1251')
windows_df.to_csv("olaf_Windows-1251.csv")
utf_df = pd.read_fwf("../input/character-encoding-examples/shisei_UTF-8.txt")
utf_df.to_csv("shisei_UTF-8.csv")
