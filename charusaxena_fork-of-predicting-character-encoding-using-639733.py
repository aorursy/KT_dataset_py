# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re
print(os.listdir("../input"))
import chardet
texts = os.listdir("../input")


# Any results you write to the current directory are saved as output.
text1 = open("../input/harpers_ASCII.txt")
text1.read()
def detect_encoding(file):
    with open(file,'rb') as raw_data:
        result = chardet.detect(raw_data.read())
        return(result)
detect_encoding("../input/harpers_ASCII.txt")
import glob
for f in glob.glob("../input/*.txt"):
    #to get filename like getting 'shisei_UTF-8.txt'
    name = os.path.basename(f)
    #to get the after "_" part i.e 'UTF-8.txt' and before "." ie to get the encoding "UTF-8"
    actual_encoding = name.split("_")[1].split(".")[0]
    actual_encoding =''.join(e for e in actual_encoding if e.isalnum())
    #to get the predicted encoding
    results = detect_encoding(f)
    predicted_encoding=results['encoding']
    predicted_encoding =''.join(e for e in predicted_encoding if e.isalnum())
    
    #to see if confidence is more than 50%
    if results['confidence']>0.50:
        if actual_encoding.lower()==predicted_encoding.lower():
            print("Correct Prediction for ",name+" given as "+actual_encoding)
        else:
            print("Incorrect Prediction for",name+" was supposed to be: ",actual_encoding +" not "+predicted_encoding)
    else:
        print("\n\nEncoding result had confidence less than 50%",predicted_encoding['encoding'])
