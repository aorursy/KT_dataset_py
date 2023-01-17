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
print(type(before))
print(f"The string before encoding is :- '{before}'")
# encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors = "replace")

# check the type
print(type(after))
print(f"The string before encoding is :- {after}")
# convert it back to utf-8
print(after.decode("utf-8"))
# try to decode our bytes with the ascii encoding
###print(after.decode("ascii"))
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
input_list = ['$', '#', '你好', 'नमस्ते']
encoded_list = [item.encode('ascii', errors='replace') for item in input_list]
print(f"The ASCII encoded input_list is: {encoded_list}")
erroneous_result = [item.decode('ascii') for item in encoded_list]
print(f"The ASCII format of the strings above is:- {erroneous_result}")
print()
encoded_list = [item.encode('utf-8', errors='replace') for item in input_list]
print(f"The UTF-8 encoded input_list is: {encoded_list}")
correct_result = [item.decode('utf-8') for item in encoded_list]
print(f"The UTF-8 format of the strings above is:- {correct_result}")
# try to read in a file not in UTF-8
###kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")
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
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as data:
    file_type = chardet.detect(data.read())
print(file_type)
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("Police_Killings_UTF-8_Format.csv")