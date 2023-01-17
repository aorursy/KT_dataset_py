# modules we'll use
import pandas as pd
import numpy as np

# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
# Let's take a look at the Kickstarter project from 2018, since we will be working with it later on in this exercise.
kickstarter_df = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv')

# We see that there are several columns with `dtype = object`
kickstarter_df.info()
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

before = 'This is a dollar sign: $'
after = before.encode('ascii', errors = 'replace')
print(after.decode('ascii'))

before = 'This is a Hanzi character: 你好'  # I think these are called Hanzi characters. Would appreciate a correction if I'm wrong :)
after = before.encode('ascii', errors = 'replace')
print(after.decode('ascii'))
# try to read in a file not in UTF-8
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")
# look at the first ten thousand bytes to guess the character encoding
with open("../input/kickstarter-projects/ks-projects-201612.csv", 'rb') as rawdata: # 'rb' for 'read bytes'
    result_2016 = chardet.detect(rawdata.read(10000))

with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
    result_2018 = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print('Kickstart 2016: ', result_2016, '\nKickstart 2018: ', result_2018, sep='')
# read in the file with the encoding detected by chardet
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252')

# look at the first few lines
kickstarter_2016.head()
# Your Turn! Trying to read in this file gives you an error. Figure out
# what the correct encoding should be and read in the file. :)
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
print('Police Killings (10,000 lines): ', result, sep='')

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read())
print('Police Killings (full): ', result, sep='')

police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201612-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv('police-killings-utf8.csv')