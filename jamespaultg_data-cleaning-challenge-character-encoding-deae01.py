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
# below code is commented as it will error and stop further execution
#print(after.decode("ascii"))
# start with a string
before = "This is the euro symbol: €"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(
# Your turn! Try encoding and decoding different symbols to ASCII and
# see what happens. I'd recommend $, #, 你好 and नमस्ते, ஜேம்ஸ் but feel free to
# try other characters. What happens? When would this cause problems?
tamil_word = 'ஜேம்ஸ்'
print(tamil_word)
before = tamil_word.encode('utf-8')
print(before)
# Now trying to decode with an another encoding 'windows-1252', but we also try to replace characters where there is error. 
# Using 'replace' is not a best solution always, as it could change the characters and make the data unusable
print(before.decode('windows-1252',errors='replace'))
# use the correct encoding to decode, then  you get the correct results
print(before.decode('utf-8'))

# try to read in a file not in UTF-8
# below code is commented as it will error and stop further execution
#kickstarter_2016 = pd.read_csv("../input/kickstarter -projects/ks-projects-201612.csv")
# look at the first ten thousand bytes to guess the character encoding
# 'rb' in the below file indicates the file mode 'Read Binary' - reads the file just as raw bytes
with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:

    # Try reading the first 10 bytes of the file
    result = chardet.detect(rawdata.read(10))   
    
    # Try reading the first 10000 bytes of the file
    result = chardet.detect(rawdata.read(10000)) 
    
    # Try reading the first 100000 bytes of the file
    result = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result)

# read in the file with the encoding detected by chardet
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252')

# look at the first few lines
kickstarter_2016.head()
# look at the first ten thousand bytes to guess the character encoding
# 'rb' in the below file indicates the file mode 'Read Binary' - reads the file just as raw bytes
with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:

    # Try reading the first 10 bytes of the file
    result = chardet.detect(rawdata.read(10))   
    print(result)
    
    # Try reading the first 10000 bytes of the file
    result = chardet.detect(rawdata.read(10000)) 
    print(result)
    
    # Try reading the first 100000 bytes of the file
    result = chardet.detect(rawdata.read(100000))
    print(result)
# how to install the package magic in the Kaggle Kernel
#In the top right corner of the 'edit notebook' screen, click on '>' to get to the 'Settings' tab, where you can add a Custom package

# running magic from package 'python-magic'
import magic
blob = open("../input/kickstarter-projects/ks-projects-201801.csv",'rb').read()
m = magic.Magic(mime_encoding=True,keep_going=True)
encoding = m.from_buffer(blob)
print(encoding)
# Just says that it is binary, not very helpful!!
# To-Do: Should check out if there are other better options
# Your Turn! Trying to read in this file gives you an error. Figure out
# what the correct encoding should be and read in the file. :)
# below code is commented as it will error and stop further execution
#police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(1000))
    print(result)
    
    result = chardet.detect(rawdata.read(10000))
    print(result)
    
    result = chardet.detect(rawdata.read(1000000000))
    print(result)
police_shooting = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
police_shooting.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
tamil_word = 'ஜேம்ஸ்'
print(tamil_word)
before = tamil_word.encode('utf-8')
print(before)
# Now trying to decode with an another encoding 'windows-1252', but we also try to replace characters where there is error. 
# Using 'replace' is not a best solution always, as it could change the characters and make the data unusable
#print(before.decode('windows-1252',errors='surrogateescape'))
# it just hangs, doesn't seem to work
# To-Do: Analyse it further
# use the correct encoding to decode, then  you get the correct results
print(before.decode('utf-8'))
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_shooting = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
police_shooting.head()
police_shooting.to_csv('police_shooting-utf-8.csv')
with open("../input/character-encoding-examples/die_ISO-8859-1.txt", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
    print(result)
    print(result['encoding'])

str = open("../input/character-encoding-examples/die_ISO-8859-1.txt", 'r',encoding=result['encoding']).read()
print(str)
with open("../input/character-encoding-examples/harpers_ASCII.txt", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
    print(result)
    print(result['encoding'])

str = open("../input/character-encoding-examples/harpers_ASCII.txt", 'r',encoding=result['encoding']).read()
print(str)
with open("../input/character-encoding-examples/olaf_Windows-1251.txt", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
    print(result)
    print(result['encoding'])

str = open("../input/character-encoding-examples/olaf_Windows-1251.txt", 'r',encoding=result['encoding']).read()
print(str)
with open("../input/character-encoding-examples/portugal_ISO-8859-1.txt", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
    print(result)
    print(result['encoding'])

str = open("../input/character-encoding-examples/portugal_ISO-8859-1.txt", 'r',encoding=result['encoding']).read()
print(str)