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
text = input("Enter string to encrypt: ")

shift = int(input("Enter shift number: "))
tapeChar = []

tapeASCII = []

tapeFlag = []
text = list(text)

text
i = 0

while i < len(text):

      tapeChar.append(text[i])

      i += 1
i = 0

while i < len(text):

      tapeFlag.append('FALSE')

      i += 1
i = 0

while i < len(text):

    num = ord(text[i])

    tapeASCII.append(num)

    i += 1
tapeASCII
tapeChar
tapeFlag
i = 0

shift=3

while i < len(text):

    flag = tapeChar[i]

    j = 0

    while j < len(text):

        if flag == tapeChar[j] and tapeFlag[j] == 'FALSE':

            if flag == " ":

                tapeASCII[j] = 32

                tapeFlag[j] = 'TRUE'

            elif tapeChar[j].islower():

                tapeASCII[j] = (ord(tapeChar[j]) + shift - 97) % 26 + 97

                tapeFlag[j] = 'TRUE'

                j += 1

            else:

                tapeASCII[j] = (ord(tapeChar[j]) + shift - 65) % 26 + 65

                tapeFlag[j] = 'TRUE'

                j += 1

        else:

            #print("else girdi")

            j += 1

    i += 1
tapeFlag

tapeASCII
tapeChar
i = 0

while i < len(text):

    flag = tapeASCII[i]

    j = 0

    while j < len(text):

        if flag == tapeASCII[j] and tapeFlag[j] == 'TRUE':

            tapeChar[j] = chr(tapeASCII[j])

            tapeFlag[j] = 'FALSE'

            j += 1

        else:

            #print("else girdi")

            j += 1

    i += 1

tapeFlag
tapeASCII
tapeChar