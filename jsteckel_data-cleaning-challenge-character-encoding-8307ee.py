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

test1 = "I would like to make lots of $$$ #glamlife"
test2 = "what about the accents? ü, Ũ, Ú"
test3 = "I wish I spoke other languages. 你好  नमस्ते"
test1ut = test1.encode("utf-8", errors = "replace")
test2ut = test2.encode("utf-8", errors = "replace")
test3ut = test3.encode("utf-8", errors = "replace")
test1as = test1.encode("ascii", errors = "replace")
test2as = test2.encode("ascii", errors = "replace")
test3as = test3.encode("ascii", errors = "replace")

print("Three codes: str, Utf-8, Ascii")
print(test2)
print(test2ut)
print(test2as)

print("..........")
print("Three sentences of Ascii decoded to utf-8")
print(test1as.decode("utf-8")) 
#print(test2ut.decode("ascii")) #gives error
print(test2as.decode("utf-8")) 
#print(test3ut.decode("ascii")) #gives error
print(test3as.decode("utf-8"))
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

#chardet for only first 10000 lines will give 'ascii' as 100% result. But reading file as ascii gives errors.
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)

#chardet for all the data gives Windows-1252 with .73 confidence.
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read())

# check what the character encoding might be
print(result)

#Sucess!!!
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
police_killings.head()

#while looking up an ascii loading error, someone suggested 'latin-1' which also works.
police_killingsl = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='latin-1')
police_killingsl.head()


#try to make diff:
police_killingsut = set(zip(police_killings))
police_killingsut8 = police_killingsut.encode("utf-8", errors = "replace")

diff = set(zip(police_killings)) - set(zip(police_killingsut))
list(diff)
#no difference detected. (need to test that this would actually do what I want)
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("police_killings-201801-utf8.csv")