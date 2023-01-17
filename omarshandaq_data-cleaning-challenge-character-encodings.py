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

# you can check that the language chinese and arabic worked the UTF-8 encode it and decode it without any problems, but when you try to encode a symbol 
# it's kinda complicated because in the last example ("This is the euro symbol: €") it worked but when i tried the dollar symbol it didn't worked
# also with # symbol, the UTF-8 doesn't encode any of those and i tried many other symbols it didn't work either as you see in the next example
test_utf = "this is a chinese language utf-8: नमस्ते"
after_test_utf = test_utf.encode("utf-8", errors = "replace")
print(after_test_utf)
print(after_test_utf.decode("UTF-8"))

symbol_test = "this is the dollar symbol: مرحبا"
a_symbol_test = symbol_test.encode("utf-8", errors = "replace")
print(a_symbol_test)
print(a_symbol_test.decode("utf-8"))

# NEXT EXAMPLE
symbol_test = "this is the US dollar symbol: $"
after_symbol_test = symbol_test.encode("utf-8", errors = "replace")
print(after_symbol_test)
print(after_symbol_test.decode("utf-8"))
#i tried to put a symbol or another language like chinese or arabic it didn't work the problem is when you try to put any symbol the ascii encode it
# and doesn't decode it 
# also the same this with the language like arabic or chinese the ascii encode it but doesn't decide it and you can see at this examble and the next one
test_ascii = "this is a chinese language: مثال"
after_test_ascii = test_ascii.encode("ascii", errors="replace")
print(after_test_ascii)
print(after_test_ascii.decode("ascii"))
test_ascii_symbol = "This is the euro symbol: €"
after_test_ascii_symbol = test_ascii_symbol.encode("ascii", errors="replace")
print(after_test_ascii_symbol)
print(after_test_ascii_symbol.decode("ascii"))
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
# this code has and error so we'll see what's the encode type
# the encoding type is 'Windows-1252' and the confidence is 0.72 but we must make the rawdata.read 100000 more than the last example
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata: result = chardet.detect(rawdata.read(100000))
print(result)
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("PoliceKillingsUS.csv")