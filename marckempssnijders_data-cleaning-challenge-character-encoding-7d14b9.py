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

# Wrong character encodings will cause problems when evaluating string contents on data frames, e.g movie sentiment reviews
before_1 = "This is the dollar symbol: $"

#encode to ascii
after_1 = before_1.encode("ascii", errors = "replace")
print( "Before: " + before_1)
print("Type after: "+ str(type(after_1)))
print("After decode: " + after_1.decode("ascii"))

before_2 = "Chinese character 你好";
after_2 = before_2.encode("ascii", errors = "replace")
print( "Before: " + before_2)
print("Type after: "+ str(type(after_2)))
print("After decode: " + after_2.decode("ascii"))

before_3 = "??? character नमस्ते";
after_3 = before_3.encode("ascii", errors = "replace")
print( "Before: " + before_3)
print("Type after: "+ str(type(after_3)))
print("After decode: " + after_3.decode("ascii"))
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
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    charEnc = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(charEnc)

#Alhough the output suggests that the characterencoding is 'ascii' it turns out to be 'latin1'
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='latin1')
police_killings.head()
# save our file (will be saved as UTF-8 by default!)
#police_killings.to_csv("PoliceKillingsUS.csv", encoding="UTF-8")

# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
# look at the first ten thousand bytes to guess the character encoding
import ftfy

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    charEnc = chardet.detect(rawdata.read(10000))
    #We will use ftfy (See below 'More practice') to print out the encoding 
    #of the first 10 character
    ftfy.explain_unicode(str(rawdata.read(10)))

# check what the character encoding might be
print(charEnc)

#Alhough the charEnc output suggests that the characterencoding is 'ascii' 
#it turns out to be 'latin1', as suggested by the ftfy module
#Although ftfy provides a fix_encoding method to automatically correct the encoding we will 
#only use the encoding it detects here and not perform ftfyautomatic conversion
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='latin1')
# save our file (will be saved as UTF-8 by default!)
police_killings.to_csv("PoliceKillingsUS_utf_8.csv", encoding="UTF-8")