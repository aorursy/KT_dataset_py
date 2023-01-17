# modules we'll use
import pandas as pd
import numpy as np

# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
# look at code here
# start with a string
before = "This is the euro symbol: €"

# check to see what datatype it is
type(before)

#print it out
print(before)
# encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors = "replace")

# check the type
type(after)

# print it out
print(after)
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
a = '$'
b = '#'
c = '你好'
d = 'नमस्ते'
# print them out
print(a)
print(b)
print(c)
print(d)
# now try for ascii
a_a = a.encode("ascii", errors = "replace")
b_a = b.encode("ascii", errors = "replace")
c_a = c.encode("ascii", errors = "replace")
d_a = d.encode("ascii", errors = "replace")
print(a_a.decode("ascii"))
print(b_a.decode("ascii"))
print(c_a.decode("ascii"))
print(d_a.decode("ascii"))
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
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)
# wrong! it is ISO-8859-1 otherwise known as latin
# BeautifulSoup is another thing we can try to automatically handle it for us

# note that we can use result directly (if correct)
#police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding=result['encoding'])
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='ISO-8859-1')
police_killings.sample(10)
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("police_killings_demongolem_utf8.csv")
# read in files here
# ../input/encoding-tests/die_ISO-8859-1.txt
with open("../input/encoding-tests/die_ISO-8859-1.txt", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)

dieset = open('../input/encoding-tests/die_ISO-8859-1.txt', encoding=result['encoding']).read()
# ../input/encoding-tests/harpers_ASCII.txt
with open('../input/encoding-tests/harpers_ASCII.txt', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)

harpersset = open('../input/encoding-tests/harpers_ASCII.txt', encoding=result['encoding']).read()
# ../input/encoding-test/olaf_Windows-1251.txt
with open('../input/encoding-tests/olaf_Windows-1251.txt', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
    
# check what the character encoding might be
print(result)

olafset = open('../input/encoding-tests/olaf_Windows-1251.txt', encoding=result['encoding']).read()
# ../input/encoding-test/portugal_ISO-8859-1.txt
with open('../input/encoding-tests/portugal_ISO-8859-1.txt', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
    
# check what the character encoding might be
print(result)

portgualset = open('../input/encoding-tests/portugal_ISO-8859-1.txt', encoding=result['encoding']).read()
# ../input/encoding-test/shisei_UTF-8.txt
with open('../input/encoding-tests/shisei_UTF-8.txt', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
    
# check what the character encoding might be
print(result)

portgualset = open('../input/encoding-tests/shisei_UTF-8.txt', encoding=result['encoding']).read()
# ../input/encoding-test/yan_BIG-5.txt
with open('../input/encoding-tests/yan_BIG-5.txt', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
    
# check what the character encoding might be
print(result)

yanset = open('../input/encoding-tests/yan_BIG-5.txt', encoding=result['encoding']).read()