# modules we'll use
import pandas as pd
import numpy as np

# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
# start with a string
before = "This is the dollar symbol: $"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))
# start with a string
before = "This is the hash symbol: #"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))
# start with a string
before = "This is the copyright symbol: ©"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))
# start with a string
before = "This is the micro sign: µ"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))
police_killings = pd.read_csv("../input/PoliceKillingsUS.csv")
# look at the first ten thousand bytes to guess the character encoding
with open("../input/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)
# read in the file with the encoding detected by chardet
policeKillings = pd.read_csv("../input/PoliceKillingsUS.csv", encoding='ascii')

# look at the first few lines
policeKillings.head()
# try with 100,000 bytes to see if we can correctly predict encoding
with open("../input/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result)
# read in the file with the encoding detected by chardet
policeKillings = pd.read_csv("../input/PoliceKillingsUS.csv", encoding='Windows-1252')

# look at the first few lines
policeKillings.head()
# save our file (will be saved as UTF-8 by default!)
policeKillings.to_csv("policeKillingsCb-utf8.csv")