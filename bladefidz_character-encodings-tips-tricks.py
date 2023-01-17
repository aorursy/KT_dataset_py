# example of decoding & re-encoding

# read in file (automatically converted to Unicode 8)
with open("../input/yan_BIG-5.txt", encoding="big5") as f:
    # read in 5000 bytes from our text file
    lines = f.readlines(5000)

# check out the last line
last_line = lines[len(lines) - 1]
print("In unicode: ", last_line)

# write out just the last line in the original encoding
# make sure you open the file in binary mode (the "b" in "wb")
with open("big5_output.txt", "wb") as f:
    # convert back to big5 as we write out our file
    f.write(last_line.encode("big5"))

# take a look to see how the encoding changes our file
print("In BIG-5: ", last_line.encode("big5"))
print(last_line)
print() # print a blank line
print("Actual length:", len(last_line))
print("Length with wrong encoding:", len(last_line.encode("big5")))
# start with a string
before = "€"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("big5", errors = "replace")

# convert it back to utf-8
print(after.decode("big5"))

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(
# import a library to detect encodings
import chardet
import glob

# for every text file, print the file name & guess its file encoding
print("File".ljust(45), "Encoding")
for filename in glob.glob('../input/*.txt'):
    with open(filename, 'rb') as rawdata:
        result = chardet.detect(rawdata.read())
    print(filename.ljust(45), result['encoding'])
# function to test if a file is in unicode
def is_it_unicode(filename):
    with open(filename, 'rb') as f:
        encoding_info = chardet.detect(f.read())
        if "UTF-8" not in encoding_info['encoding']: 
            print("This isn't UTF-8! It's", encoding_info['encoding'])
        else: 
            print("Yep, it's UTF-8!")

# test our function, the first one is not unicode, the second one is!
is_it_unicode("../input/die_ISO-8859-1.txt")
is_it_unicode("../input/shisei_UTF-8.txt")
# import the "fixed that for you" module
import ftfy

# use ftfy to guess what the underlying unicode should be
print(ftfy.fix_text("The puppyÃ¢â‚¬â„¢s paws were huge."))
# use ftfy to guess what the underlying unicode should be
print(ftfy.fix_text("&macr;\\_(ã\x83\x84)_/&macr;"))