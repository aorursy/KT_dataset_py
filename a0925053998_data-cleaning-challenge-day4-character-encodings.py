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
test_code = "$, #, 你好, नमस्ते, A, a, 10, 10.0, @, ~"
after_asc = test_code.encode("ascii", errors = "replace")
#after_asc
print(after_asc.decode("ascii"))
kickstarter_201801 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")
kickstarter_201801.head()
# try to read in a file not in UTF-8
kickstarter_201612 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")
# look at the first ten thousand bytes to guess the character encoding
with open("../input/kickstarter-projects/ks-projects-201612.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)
# read in the file with the encoding detected by chardet
kickstarter_201612 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252')

# look at the first few lines
kickstarter_201612.head()
# Your Turn! Trying to read in this file gives you an error. Figure out
# what the correct encoding should be and read in the file. :)
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdatapk1:
    result_pk = chardet.detect(rawdatapk1.read(50000))

print(result_pk)
PoliceKillingsUS_U = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding="Windows-1252")

PoliceKillingsUS_U.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_201612.to_csv("ks-projects-201612-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
PoliceKillingsUS_U.to_csv("PoliceKillingsUS-utf8.csv")