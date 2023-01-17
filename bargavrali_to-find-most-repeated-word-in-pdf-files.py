# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#
# Hi, I am practising the Data Science Concepts and below is the Program, that 
# reads text from multiple Images and Computes to find the mosted repeated word of 
# all those files.
#
import glob 
import wand
import pandas as pd
from wand.image import Image
from PIL import Image as img1
import pytesseract

pdfs_path=glob.glob('../input/filesdata/*.pdf')

pdfs_info=[]
words_dic={}
high_prob_word=''
high_prob_word_count=0

for path in pdfs_path:
    with(Image(filename=path,resolution=120)) as source:
        for pg_no,image in enumerate(source.sequence):
            newfilename = "/kaggle/working/" + path.split("/")[-1][:-4] +  "_" + str(pg_no + 1) + '.jpeg'
            Image(image).save(filename=newfilename)
            page_text = pytesseract.image_to_string(img1.open(newfilename))
            print(path.split("/")[-1], newfilename[-6])
            pdfs_info.append([path.split("/")[-1], newfilename[-6] , page_text])
for data in pdfs_info:
    for word in data[2].split():
        words_dic[word] = words_dic.setdefault(word,0) + 1

for word_data in words_dic.items():
    word=word_data[0]
    word_count=word_data[1]
    if word_count == max(words_dic.values()):
        high_prob_word = word
        high_prob_word_count=word_count
print("The Most repeated Word is \'",high_prob_word,"\'. It is repeated for",high_prob_word_count,
      "times over all the files with probability of",round(high_prob_word_count/sum(words_dic.values()),2))