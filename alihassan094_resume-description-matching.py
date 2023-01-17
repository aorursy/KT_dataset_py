# Matching Resume and the Description
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
file_names=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file1 = os.path.join(dirname, filename)

        file_names.append(file1)

file_names.sort()

file_names
len(file_names)
test_file = file_names[10]

test_file
# pip install docx2txt
# pip install PyMuPDF
import docx2txt

import sys, fitz
fname = file_names[0]

fname
# fname = '/kaggle/input/AdamCaplanResume2020 (1).pdf'

fname = test_file

# fname = '/kaggle/input/resume-dataset/Ali Hassan_Resume.pdf'

doc = fitz.open(fname)

doc_text = ""

for page in doc:

    doc_text = doc_text + str(page.getText())

    

# doc_text
# resume_charles = docx2txt.process(file_names[0])

resume_charles = doc_text

resume_ali = docx2txt.process(file_names[3])

# resume = docx2txt.process('1569871831-8874Matthew Kleifges Resume .docx')

# print(resume)
file_names[8]
# job_description = docx2txt.process('Data and Investments VP.docx')

jd = file_names[8][14:]

job_description = docx2txt.process(file_names[8])
joint_1 = [resume_charles, job_description]

joint_2 = [resume_ali, job_description]
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

# count = cv.fit_transform(joint)

count_charles = cv.fit_transform(joint_1)

count_ali = cv.fit_transform(joint_2)
from sklearn.metrics.pairwise import cosine_similarity

# cs = cosine_similarity(count)

cs_charles = cosine_similarity(count_charles)

cs_ali = cosine_similarity(count_ali)

# print(cs)
print('The job description is', jd)

print(test_file[14:], cs_charles[0][1]*100, ' percent')

print('Ali CV matches ', cs_ali[0][1]*100, ' percent')