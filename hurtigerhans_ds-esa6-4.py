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

after
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

# try other characters. What happen? When would this cause problems?

drawling_lang = "नमस्ते"

dl_after=drawing_lang.encode("ascii", errors="replace")

dl_after
dl_original=dl_after.decode("ascii")

dl_original
dl0 = "नमस्ते"

dl0_after=dl0.encode("utf-8", errors="replace")

dl0_after
dl0_original=dl_after.decode("utf-8")

dl0_original
dl1 = "!§$@"

dl1a = dl.encode("ascii", errors="replace")

dl1a
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
with open ("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv","rb") as rdata:

    dataenc = chardet.detect(rdata.read(100000))

    print(dataenc)

    

#with using "dataenc = chardet.detect(rdata.read(10000))" (one 0 less) chardet gives as encoding type ascii with a confidence of 1.0 (Screenshot in pdf)
police_killing = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding="Windows-1252")
# save our file (will be saved as UTF-8 by default!)

kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 

police_killing.to_csv("police_killing.csv")
