import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import io # I/O streams

import codecs # string encoding & decoding

import this
str = codecs.decode(this.s,'rot13') # decode text

df = pd.read_csv(io.StringIO(str),delimiter='|') # read in DataFrame

df.to_csv('this.csv',index = False) # output to CSV