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
print(after)
# check the type
type(after)
# take a look at what the bytes look like
after
# convert it back to utf-8
print(after.decode("UTF-8"))

# if we don't use correct encoding for decoding the string we will get an error
print(after.decode("ASCII"))
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

s1 = '$'
s2 = '#'
s3 = '你好'
s4 = 'नमस्ते'
s5 = '`Hello'

encoded_1 = s1.encode('UTF-8',errors='replace')
encoded_2 = s2.encode('UTF-8',errors='replace')
encoded_3 = s3.encode('UTF-8',errors='replace')
encoded_4 = s4.encode('UTF-8',errors='replace')
encoded_5 = s5.encode('UTF-8',errors='replace')

ascii_encoded_1 = s1.encode('ASCII',errors = 'replace')
ascii_encoded_2 = s2.encode('ASCII',errors = 'replace')
ascii_encoded_3 = s3.encode('ASCII',errors = 'replace')
ascii_encoded_4 = s4.encode('ASCII',errors = 'replace')
ascii_encoded_5 = s5.encode('ASCII',errors = 'replace')

print("UTF-8 encoded: {0}  ASCII encoded: {1}".format(encoded_1.decode('UTF-8'),ascii_encoded_1.decode('ASCII')))
print("UTF-8 encoded: {0}  ASCII encoded: {1}".format(encoded_2.decode('UTF-8'),ascii_encoded_2.decode('ASCII')))
print("UTF-8 encoded: {0}  ASCII encoded: {1}".format(encoded_3.decode('UTF-8'),ascii_encoded_3.decode('ASCII')))
print("UTF-8 encoded: {0}  ASCII encoded: {1}".format(encoded_4.decode('UTF-8'),ascii_encoded_4.decode('ASCII')))
print("UTF-8 encoded: {0}  ASCII encoded: {1}".format(encoded_5.decode('UTF-8'),ascii_encoded_5.decode('ASCII')))
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

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",'rb') as rawdata:
    encoding_result = chardet.detect(rawdata.read(500000))
print(encoding_result)

police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding='Windows-1252')
police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("police_killings_utf8.csv")