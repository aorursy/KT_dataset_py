# modules we'll use

import pandas as pd

import numpy as np



# helpful character encoding module

import chardet



# set seed for reproducibility

np.random.seed(0)
police_killings_us = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")

police_killings_us.sample(10)
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



str = "$#§$%&/()=? 你好 नमस्ते вогачир لعاعغ 🤠"



str_ascii_replace = str.encode("ascii", errors = "replace")

print(str_ascii_replace.decode("ascii")) # ==> $#?$%&/()=? ?? ?????? ??????? ????? ?



str_ascii_ignore = str.encode("ascii", errors = "ignore")

print(str_ascii_ignore.decode("ascii")) # ==> $#$%&/()=?     



str_koi8_r_replace = str.encode("koi8_r", errors = "replace")

print(str_koi8_r_replace.decode("koi8_r")) # ==> $#?$%&/()=? ?? ?????? вогачир ????? ?



str_utf_16_replace = str.encode("utf_16", errors = "replace")

print(str_utf_16_replace.decode("utf_16")) # ==> $#§$%&/()=? 你好 नमस्ते вогачир لعاعغ 🤠

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

    my_result = chardet.detect(rawdata.read(100000))



police_killings_us = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding=my_result["encoding"])

police_killings_us.sample(10)
# save our file (will be saved as UTF-8 by default!)

kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 



police_killings_us.to_csv("PoliceKillingsUS-UTF8.csv")