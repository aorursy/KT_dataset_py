# modules we'll use

import pandas as pd

import numpy as np



# helpful character encoding module(WHich can help us detect the encoding for a single file or for multiple files)

#in succession



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



'''The problem with chinese and Hindi characters would be that they are not a part of standard



UTF-8 encodings as they have their own specific encodings whic cannot be interpreted if we just try to use 

the standard UTF-8 encoding'''



dummyStr="नमस्ते"



encodedStr=dummyStr.encode('ascii',errors='replace')



print(encodedStr)

print(encodedStr.decode("ascii"))
# try to read in a file not in UTF-8

kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")
# look at the first ten thousand bytes to guess the character encoding(Reading as a binary string as

#detect expects data in bytes)



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



'''Since the error message was about a particular character not being parsed,

on a simple google search revealed that this is a part of the Latin 1 character set.Using this definitely works



In this case though,the chardlet guess is wrong as it suggests that the character encoding is in ascii but actually

the file is encoded in Latin-1 encoding

'''



with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",'rb') as data:

    result = chardet.detect(data.read(10000))



print(result)



police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding="Latin-1")
# save our file (will be saved as UTF-8 by default!)

kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding

police_killings.to_csv("police_killings_correctEncoding.csv")