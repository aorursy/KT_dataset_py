import numpy as np # linear algebra

import pandas as pd # data processing



import chardet # helpful for character encoding



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# start with a string

before = "This is the euro symbol : €"



# check to see what datatype it is

type(before)
# encode it to a different encoding, replacing characters that raise errors

after = before.encode("utf-8", errors = "replace")



# check the type

type(after)
after
# convert it back to utf-8

after.decode("utf-8")
# try to decode our bytes with the ascii encoding

after.decode("ascii")
# start with a string

before = "This is the euro symbol : €"



# encode it to a different encoding, replacing characters that will raise errors

after = before.encode("ascii", errors = "replace")



# convert it back to utf-8

print(after.decode("ascii"))
# try to read in a file not in UTF-8

kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")
# look at the first 10000 bytes to guess the character encoding

with open("/kaggle/input/kickstarter-projects/ks-projects-201801.csv", "rb") as rawdata:

    result = chardet.detect(rawdata.read(10000))



# check what the character encoding might be

print(result)
# read in the file with the encoding detected by chardet

kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding = "Windows-1252")



# look at the first few lines

kickstarter_2016.head()
# try on a different dataset

police_killing = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")
# look at th first 10000 bytes to guess the character encoding

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", "rb") as rawdata:

    result = chardet.detect(rawdata.read(100000))

    

# check what the character encoding might be

print(result)
# read in the file with the character encoding detected by chardet

police_killing = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding = "Windows-1252")



# looking at the first few rows

police_killing.head()
# save our file (saved as UTF-8 by default!)

kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")