# import a library to detect encodings

import chardet

import glob



# for every text file, print the file name & a gues of its file encoding

print("File".ljust(45), "Encoding")

for filename in glob.glob('../input/*.txt'):

    with open(filename, 'rb') as rawdata:

        result = chardet.detect(rawdata.read())

    print(filename.ljust(45), result['encoding'])
# function to test if a file is in unicode

def isItUnicode(filename):

    with open(filename, 'rb') as f:

        encodingInfo = chardet.detect(f.read())

        if "UTF" not in encodingInfo['encoding']: 

            print("This isn't Unicode! It's", encodingInfo['encoding'])

        else: 

            print("Yep, it's Unicode.")

 

# test our function, the first one is not unicode, the second one is!

isItUnicode("../input/die_ISO-8859-1.txt")

isItUnicode("../input/shisei_UTF-8.txt")