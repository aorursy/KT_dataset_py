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
# $, #, 你好 and नमस्ते 
# start with a string
before_dollar = "This is the USD symbol: $"
before_hash = "This is the hash symbol: #"
before_sym = "This is the sym symbol: 你好"
before_नमस्ते = "This is the hindi symbol: नमस्ते"

# check to see what datatype it is
type(before_dollar)
type(before_hash)
type(before_sym)
type(before_नमस्ते)

# encode it to a different encoding, replacing characters that raise errors
after_dollar = before_dollar.encode("utf-8", errors = "replace")
after_hash = before_hash.encode("utf-8", errors = "replace")
after_sym = before_sym.encode("utf-8", errors = "replace")
after_नमस्ते = before_नमस्ते.encode("utf-8", errors = "replace")

# check the type
type(after_dollar)
type(after_hash)
type(after_sym)
type(after_नमस्ते)

# take a look at what the bytes look like
after_dollar
after_hash
after_sym
after_नमस्ते


# convert it back to utf-8
print(after_dollar.decode("utf-8"))
print(after_hash.decode("utf-8"))
print(after_sym.decode("utf-8"))
print(after_नमस्ते.decode("utf-8"))

# start with a string
before_dollar = "This is the USD symbol: $"
before_hash = "This is the USD symbol: #"
before_sym = "This is the USD symbol: 你好"
before_नमस्ते = "This is the USD symbol: नमस्ते"


# encode it to a different encoding, replacing characters that raise errors
after_dollar = before_dollar.encode("ascii", errors = "replace")
after_hash = before_hash.encode("ascii", errors = "replace")
after_sym = before_sym.encode("ascii", errors = "replace")
after_नमस्ते = before_नमस्ते.encode("ascii", errors = "replace")

# convert it back to ascii
print(after_dollar.decode("ascii"))
print(after_hash.decode("ascii"))
print(after_sym.decode("ascii"))
print(after_नमस्ते.decode("ascii"))



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
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata_police:
    result_police = chardet.detect(rawdata_police.read(10000))

# check what the character encoding might be
print(result_police)
# though it says ascii it fails to read in ascii, tried utf-8 also finally able to read file with Windows-1252
# read in the file with the encoding detected by chardet
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')

# look at the first few lines
police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("pc-killings-201803-utf8.csv")