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

#lets try $ first

before = "this is $"
type(before)
#now encode it to utf-8
after = before.encode("utf-8", errors = "replace")

type(after)
#let's see what after looks after encoding into utf-8
#but it didn't effect the $ here because its in its range of special characters
after
#lets convert it to ascii
#we will see that $ is also not effected in ascii
after = before.encode("ascii", errors = "replace")

after
#if decode the ascii into utf-8 
#if it is encoded properly it won't effect the data

before = after.decode("ascii")

before
#so now we will try hindi

before = "नमस्ते"

#encoding

after = before.encode("utf-8", errors = "replace")
#printing the stuff after encoding 
#what we will get is mojibake value
after
#decoding & printing 
before = after.decode("utf-8")

before
#now lets try to decode the utf-8 value into ascii 
# what we will get is error
before = after.decode("ascii")

before
#now lets encode it into ascii & try

before = "नमस्ते"

after = before.encode("ascii", errors = "replace")

after
#now lets decode & check whether we get नमस्ते or not
#no we won't get so, that's why utf-8 is better 
before = after.decode("ascii")
before
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
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(1000))

result
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='ascii')

police_killings.head()
#since it shows errors even after 100% confidence 
#we will take more data this time to detect

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(1000000))

result
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')

police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 

police_killings.to_csv("PoliceKillingsUS.csv")