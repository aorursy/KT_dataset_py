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
# start with a string
before1 = "This is the dollar symbol: $"
before2 = "This is the # symbol"
before3 = "This is 你好 "
before4 = "नमस्ते"

# encode it to a different encoding, replacing characters that raise errors
after1 = before1.encode("ascii", errors = "replace")
after2 = before2.encode("ascii", errors = "replace")
after3 = before3.encode("ascii", errors = "replace")
after4 = before4.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after1.decode("ascii"))
print(after2.decode("ascii"))
print(after3.decode("ascii"))
print(after4.decode("ascii"))
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
# look at the first ten thousand bytes to guess the character encoding
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)
# Your Turn! Trying to read in this file gives you an error. Figure out
# what the correct encoding should be and read in the file. :)
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding = "Windows-1252")
police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("police_killings_201803-utf8.csv")
import chardet
import glob
import ftfy
import os
# for every text file, print the file name & a gues of its file encoding
print("File".ljust(60), "Encoding")
for filename in glob.glob('../input/character-encoding-examples/*.txt'):
    rawdata = open(filename, 'rb').read()
    result = chardet.detect(rawdata)
    print(filename.ljust(60), result['encoding'])
files = os.listdir("../input/character-encoding-examples")
def detect_encoding(file, sample_sizes=[1000, 100000]):
    with open(f'../input/character-encoding-examples/{file}', 'rb') as raw_file:
        encoding = None
        for size in sample_sizes:
            enc = chardet.detect(raw_file.read(size))
#             encoding = enc if encoding is None else (enc if enc['confidence'] > encoding['confidence'] else encoding)
            encoding = enc if enc['encoding'] is not None else encoding
    
    return encoding


encodings = dict()
for file in files:
    encoding = detect_encoding(file)
    encodings[file] = encoding
    
print(encodings)
for file in files:
    print(file)
    with open(f'../input/character-encoding-examples/{file}', encoding=encodings[file]['encoding'], mode='r') as f:
        lines = f.readlines()
        fixed_lines = [ftfy.fix_text(line) for line in lines]
    
with open("../input/character-encoding-examples/file_guide.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be

print(result)
file_guide = pd.read_csv("../input/character-encoding-examples/file_guide.csv") #, encoding = "utf-8")
file_guide.head()