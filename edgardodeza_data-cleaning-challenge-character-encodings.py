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
# original string
print(before)

# Let's encode in UTF-8
encode_utf = before.encode("utf-8", errors = "replace")
print(encode_utf)

# Let's decode it back using UTF-8
decode_encode_utf = encode_utf.decode('UTF-8')
print(decode_encode_utf)
# Your turn! Try encoding and decoding different symbols to ASCII and
# see what happens. I'd recommend $, #, 你好 and नमस्ते but feel free to
# try other characters. What happens? When would this cause problems?

dollar = '$'
sharp = '#'
chinese = '你好'
hindi = 'नमस्ते'
# let's write a function that encodes a string 'my_string' using 
# the encoding 'encoding'.
# We check the type,
# we decode back to a string,
# we check the type again.

def encode_decode_test(my_string, encoding):
    # original character
    print("original character: ", my_string)
    print("type: ", type(my_string))
    print()

    # encode string
    my_string_encode = my_string.encode(encoding, errors = "replace")
    print("encoded: ", my_string_encode)
    print("type: ", type(my_string_encode))
    print()

    # let's decode it back to string
    my_string_encode_decode = my_string_encode.decode(encoding, errors = "replace")
    print("decode back to string: ", my_string_encode_decode)
    print("type: ", type(my_string_encode_decode))
encode_decode_test(dollar, 'utf-8')
encode_decode_test(sharp, 'utf-8')
encode_decode_test(chinese, 'utf-8')

encode_decode_test(hindi,  'utf-8')
encode_decode_test(dollar, 'ascii')
encode_decode_test(sharp, 'ascii')
encode_decode_test(chinese, 'ascii')
encode_decode_test(hindi, 'ascii')
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
police_killings_US = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")
police_killings_US.head()
# look at the first ten thousand bytes to guess the character encoding
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)
police_killings_US = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='ascii')
# look at the first ten thousand bytes to guess the character encoding
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(80000))

# check what the character encoding might be
print(result)
# look at the first ten thousand bytes to guess the character encoding
police_killings_US = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
police_killings_US.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings_US.to_csv("PoliceKillingsUS-utf8.csv")