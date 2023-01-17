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
#   # start with a string
before = "This symbol 'नमस्ते' is dope"
#  # check the type
type(before)
#  # encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")
type(after)#bytes
# # convert it back to utf-8
print(after.decode("ascii")) ##This symbol '??????' is dope
#  # encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors = "replace")
#  # check the type
type(after)#bytes
#  # convert it back to utf-8
print(after.decode("utf-8"))##This symbol 'नमस्ते' is dope


# try other characters. What happens? When would this cause problems?
#   # start with a string
before = "This symbol '你好' is not so dope"
#  # check the type
type(before)
#  # encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")
type(after)#bytes
#  # encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors = "replace")
#  # check the type
type(after)#bytes
#  # convert it back to utf-8
print(after.decode("utf-8"))###This symbol '你好' is not so dope
#  # encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")
type(after)#bytes
# # convert it back to utf-8
print(after.decode("ascii")) ###This symbol '??' is not so dope
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


#   #https://khanyisasolutions.co.za/

#  # look at the first ten thousand bytes to guess the character encoding
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result) ## it says it's ASCII with 100% confidence({'encoding': 'ascii', 'confidence': 1.0, 'language': ''})


#  # read in the file with the encoding detected by chardet
#police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='ascii')# it said ascii but ascii doesn't work (ERROR:- UnicodeDecodeError: 'ascii' codec can't decode byte 0x96 in position 2: ordinal not in range(128))
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')#This one works(????)***
# look at the first few lines
police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("fatal_police_shootings_us_201804.csv")