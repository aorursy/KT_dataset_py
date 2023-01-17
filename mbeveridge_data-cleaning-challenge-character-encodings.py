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

# ---
# MB: EXAMPLE 1
# start with a string
before = "This is the dollar symbol: $"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))
# "What happens?" : `This is the dollar symbol: $` ...This has worked because ASCII has a symbol for the dollar

# ---
# MB: EXAMPLE 2
# start with a string
before = "This is the Chinese symbol for hello: 你好"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))
# "What happens?" : `This is the Chinese symbol for hello: ??` ...This has NOT worked, because ASCII has NO Chinese symbols
# ...(and so it was incorrect to tell `encode()` that the original `before` text was written in ASCII)

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

# ---
# MB: ATTEMPT 1
# `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x96 in position 2: invalid start byte`
# SO, comment-out the code above, and use `chardet` :

# look at the first ten thousand bytes to guess the character encoding
# with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
#    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
# print(result)
# `{'encoding': 'ascii', 'confidence': 1.0, 'language': ''}` --> ASCII, with 100% confidence !

# read in the file with the encoding detected by chardet
# police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='ascii')

# look at the first few lines
# police_killings.head()
# `UnicodeDecodeError: 'ascii' codec can't decode byte 0x96 in position 2: ordinal not in range(128)` ...SO that's weird !

# ---
# MB: ATTEMPT 2
# SO, comment-out the code above, and try again :

# look at the first ONE HUNDRED thousand bytes to guess the character encoding
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result)
# `{'encoding': 'Windows-1252', 'confidence': 0.73, 'language': ''}` --> 'Windows-1252', with 73% confidence !

# read in the file with the encoding detected by chardet
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')

# look at the first few lines
police_killings.head()

# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding

# save our file (will be saved as UTF-8 by default!)
police_killings.to_csv("PoliceKillingsUS-201803-utf8.csv")