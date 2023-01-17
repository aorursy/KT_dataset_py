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

original = 'ひらがな'

utf8_encoded = original.encode('utf-8')

utf8_encoded.decode('ascii')
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

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as data:
    result2 = chardet.detect(data.read(10000))

print(result2)
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')

police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv('PoliceKillingsUS_utf8.csv')
# read the file that describes the rest
file_guide = pd.read_csv('../input/character-encoding-examples/file_guide.csv')
file_guide
file_guide['path'] = '../input/character-encoding-examples/' + file_guide['File']
file_guide
file_guide[file_guide.Author == 'Yan Zhitui'].index[0]
results = []
for path in file_guide.path:
    with open(path, 'rb') as data:
        result = chardet.detect(data.read(50000))
        results.append(result)

file_guide['Results'] = results

file_guide
for index, row in file_guide.iterrows():
    #open the file with the right encoding
    with open (row.path, encoding=row.Results['encoding']) as f:
        string = f.read()
        text_file = open(row.File, "w")
        text_file.write(string)
        print('Wrote: ' + text_file.name)
        text_file.close()