# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import chardet

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



list_files=os.listdir('../input/')

print(list_files)
'''Read the File

Detect the encoding using chardet.detect

use that decoding to decode the text for the file(Read binary data back to Our text)

Then write this text in a .txt file using utf-8 encoding

'''



for index,name in enumerate(list_files):

    file_path='../input/'+name

    with open(file_path,'rb') as file:

        content_bytes=file.read()

        detected = chardet.detect(content_bytes)

        encoding = detected['encoding']

        

        if encoding=='utf-8' or encoding=='ascii':

            print("SKip this file")

        else:

            print(f"{name}: detected as {encoding}.")

            content_text = content_bytes.decode(encoding)

            

            new_file_path="temp_output"+str(index)+".txt"

            

            with open(new_file_path,'w') as writeFile:

                    writeFile.write(content_text)            