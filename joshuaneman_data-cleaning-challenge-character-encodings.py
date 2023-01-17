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

test1 = "$"
test2 = "#"
test3 = "你好"
test4 = "नमस्ते"
test1_encode = test1.encode("ascii", errors = "replace")
test2_encode = test2.encode("ascii", errors = "replace") 
test3_encode = test2.encode("ascii", errors = "replace") 
test4_encode = test2.encode("ascii", errors = "replace") 
print(test1_encode.decode("ascii"))
print(test2_encode.decode("ascii"))
print(test3_encode.decode("ascii"))
print(test4_encode.decode("ascii"))
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
#with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata_pk:
 #   result_pk = chardet.detect(rawdata_pk.read(100000000))
#print(result_pk)

police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("police_killings-201807-jrn-utf8.csv") 
#Open first sample file, identify encoding
with open ("../input/character-encoding-examples/die_ISO-8859-1.txt", 'rb') as file_1:
    result_1 = chardet.detect(file_1.read(10000))
print(result_1)
#Read first sample file with original encoding
file_1 = pd.read_fwf("../input/character-encoding-examples/die_ISO-8859-1.txt", encoding=result_1['encoding'])
print(file_1.head())
#Write first sample file with UTF-8 encoding
file_1.to_csv("die_UTF-8.csv")
print(file_1)