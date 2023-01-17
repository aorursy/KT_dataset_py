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
# convert a string object into a bytes object - the character encoding needs to be specified
after = before.encode("utf-8", errors = "replace")

# check the type
type(after)
# take a look at what the bytes object looks like
# for this ASCII as a character encoding is assumed - as UTF-8 and ASCII are identical with regard
# for the conventional latin characters, most of this will return the correct characters
after
# convert it back to utf-8
print(after.decode("utf-8"))
# try to decode our bytes with the ascii encoding
print(after.decode("ascii"))

# print(after.decode("ascii")) results in an UnicodeDecodeError:
# 'ascii' codec can't decode byte 0xe2 in position 25: ordinal not in range(128)
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
text1 = "I'd recommend $, #, 你好 and नमस्ते but feel free"
text1_bytes=text1.encode("ascii", errors="replace")
print(text1_bytes.decode("ascii"))

# It's clear what happens: code points which aren't in the character encoding get replaced by '?'
# Some of the exotic characters may be encoded by several bytes.
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
#police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result2 = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result2)
policeKillingsUS = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
policeKillingsUS.head()

# That's quite interesting: using the first 10000 bytes, chardet.detect returns 'ascii' with a 
# confidence of 1 - but trying to decode with ascii raises the same exception.
# Only after using 100000 bytes, the seemingly correct encoding is returned by chardet.
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
policeKillingsUS.to_csv("police_killings_us_data.csv")