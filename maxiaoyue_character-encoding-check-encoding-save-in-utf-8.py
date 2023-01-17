# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import package to detect encoding schemes
import chardet
# Clafify some concepts
before = "This is the euro symbol: €" #string = sequence of characters
print (before)
print (type(before), '\n')

after = before.encode("utf-8", errors = "replace") #bytes = sequence of integers
print (after) #printed out as characters as if encoded by ASCII
print (type(after))

print (after.decode("utf-8")) #decode bytes 
# Read file with non utf-8 encoding...error
with open('../input/portugal_ISO-8859-1.txt', 'r') as f:
    text = f.read(100000)    
print (text[-100:])
# Read file with non utf-8 encoding in binary form...success
with open('../input/portugal_ISO-8859-1.txt', 'rb') as f:
    text = f.read(100000)    
print (text[:260])
# Read in file in binary form and detect its encoding scheme
with open('../input/portugal_ISO-8859-1.txt', 'rb') as f:
    encoding = chardet.detect(f.read(100000))    
print (encoding)
# Read in file with correct encoding
with open('../input/portugal_ISO-8859-1.txt', 'r', encoding='ISO-8859-1') as f:
    text = f.read()
print (text[:260])
print (type(text))
len(text)
# Save file in utf-8(automatic)
with open ('portugal_utf-8.txt', 'w') as f:
    f.write(text)
# Check new file encoding
with open ('portugal_utf-8.txt', 'rb') as f:
    encoding = chardet.detect(f.read())
print (encoding)
with open ('portugal_utf-8.txt', 'r') as f:
    text_new = f.read(100000)
print(text_new[:260])
print (len(text_new))
print (type(text_new))
# Encode string with utf-8 explicitly
text_after = text.encode('utf-8', errors = "replace")
print (type(text_after))
# Save in utf-8 explicitly
with open('portugal_utf-8.txt', 'wb') as f:
    f.write(text_after)
with open('portugal_utf-8.txt', 'r') as f:
    text_new = f.read(100000)
print(text_new[:260])
# check encoding
with open ('../input/harpers_ASCII.txt', 'rb') as f:
    encoding = chardet.detect(f.read(10000))
print (encoding)
# read by default
with open ('../input/harpers_ASCII.txt', 'r') as f:
    text = f.read()
print (text[:260])
print (len(text))
print (type(text))
# save in utf-8 implicitly
with open ('../input/harpers_ASCII.txt', 'r', encoding = 'ascii') as f:
    text = f.read()

with open('harpers_utf-8.txt', 'w') as f:
    f.write(text)
    
with open('harpers_utf-8.txt', 'rb') as f:
    encoding = chardet.detect(f.read())
print (encoding)
# save in utf-8 explicitly
with open ('../input/harpers_ASCII.txt', 'r', encoding = 'ascii') as f:
    text = f.read()

text_bytes = text.encode('utf-8')
print (type(text_bytes))

with open('harpers_utf-8.txt', 'wb') as f:
    f.write(text_bytes)
    
with open('harpers_utf-8.txt', 'rb') as f:
    encoding = chardet.detect(f.read())
print (encoding)
# function to convert file in utf-8
def encode_with_utf8(file):
    #determine encoding
    with open('../input/'+file, 'rb') as f:
        encoding = chardet.detect(f.read())
    print ('{} original encoding is {}'.format(file, encoding['encoding']))
    
    #read in file with correct encoding
    with open('../input/'+file, 'r', encoding = encoding['encoding']) as f:
        text = f.read()
    
    #save text to new file (automatially in utf-8)
    new_file = file[:-4]+'-utf-8.txt'
    with open(new_file, 'w') as f:
        f.write(text)
    
    with open(new_file, 'rb') as f:
        encoding = chardet.detect(f.read())
    print ('{} encoding is {}'.format(new_file, encoding['encoding']))   
encode_with_utf8('portugal_ISO-8859-1.txt')
encode_with_utf8('yan_BIG-5.txt')
from mlxtend.file_io import find_files
print("Original File".ljust(35), "Original Encoding".ljust(20), "New Encoding".ljust(15), "New File Name")

for file in find_files('.txt', '../input/'):
    
    #determine encoding
    with open(file, 'rb') as f:
        encoding = chardet.detect(f.read())

    #read in file with correct encoding
    with open(file, 'r', encoding = encoding['encoding']) as f:
        text = f.read()
    
    #save text to new file (automatially in utf-8)
    new_file = file[9:-4]+'-utf-8.txt'
    with open(new_file, 'w') as f:
        f.write(text)
    
    with open(new_file, 'rb') as f:
        new_encoding = chardet.detect(f.read()) 

    print(file.ljust(35), encoding['encoding'].ljust(20), new_encoding['encoding'].ljust(15), new_file)