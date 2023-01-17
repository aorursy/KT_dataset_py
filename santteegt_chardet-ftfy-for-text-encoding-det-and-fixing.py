# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import chardet
import ftfy

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

files = os.listdir("../input")
print(files)

# Any results you write to the current directory are saved as output.
def detect_encoding(file, sample_sizes=[1000, 100000]):
    with open(f'../input/{file}', 'rb') as raw_file:
        encoding = None
        for size in sample_sizes:
            enc = chardet.detect(raw_file.read(size))
            print(f"Encoding for file {file} using sample size {size}: {enc}")
#             encoding = enc if encoding is None else (enc if enc['confidence'] > encoding['confidence'] else encoding)
            encoding = enc if enc['encoding'] is not None else encoding
    
    return encoding
encodings = dict()
for file in files:
    encoding = detect_encoding(file)
    print(f'==> Detected encoding: {encoding}')
    encodings[file] = encoding
    
print(encodings)
file = files[0]
print(file)
with open(f'../input/{file}', encoding=encodings[file]['encoding'], mode='r') as f:
    lines = f.readlines()
    print(lines)
    fixed_lines = [ftfy.fix_text(line) for line in lines]
    print(fixed_lines)
file = files[1]
print(file)
print(encodings[file])
with open(f'../input/{file}', encoding=encodings[file]['encoding'], mode='r') as f:
    print(f.readlines())
file = files[2]
print(file)
print(encodings[file])
with open(f'../input/{file}', mode='rb') as f:
    lines = ftfy.fix_file(f, encoding=encodings[file]['encoding'])
    [print(line) for line in lines]
    
file = files[3]
print(file)
print(encodings[file])
with open(f'../input/{file}', encoding=encodings[file]['encoding'], mode='r') as f:
    print(f.readlines())
file = files[5]
print(file)
print(encodings[file])
with open(f'../input/{file}', encoding=encodings[file]['encoding'], mode='r') as f:
    print(f.readlines())
file = files[6]
print(file)
print(encodings[file])
with open(f'../input/{file}', encoding=encodings[file]['encoding'], mode='r') as f:
    print(f.readlines())
