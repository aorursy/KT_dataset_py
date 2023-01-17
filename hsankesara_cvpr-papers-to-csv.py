!pip install pdfminer.six
!pip install PyPDF2
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

from wand.image import Image as Img

import io

import subprocess

from pdfminer.converter import TextConverter

from pdfminer.pdfinterp import PDFPageInterpreter

from pdfminer.pdfinterp import PDFResourceManager

from pdfminer.pdfpage import PDFPage

from tqdm import tqdm

import PyPDF2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/cvpr2019/CVPR2019/"))



# Any results you write to the current directory are saved as output.
Img(filename="../input/cvpr2019/CVPR2019/papers/Acuna_Devil_Is_in_the_Edges_Learning_Semantic_Boundaries_From_Noisy_CVPR_2019_paper.pdf", resolution=300)
!pdf2txt.py -o health.txt ../input/cvpr2019/CVPR2019/papers/Acuna_Devil_Is_in_the_Edges_Learning_Semantic_Boundaries_From_Noisy_CVPR_2019_paper.pdf
pdfTxtFile = 'health.txt'

pdf_txt = open(pdfTxtFile, 'r', encoding='utf-8')

strn = ''

# loop over all the lines

for line in pdf_txt:

    strn += line
print(strn)
!rm health.txt
pdfFileObj = open("../input/cvpr2019/CVPR2019/papers/Acuna_Devil_Is_in_the_Edges_Learning_Semantic_Boundaries_From_Noisy_CVPR_2019_paper.pdf", 'rb')

pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
print(pdfReader.numPages)
pageObj = pdfReader.getPage(0)

# extracting text from page.

# this will print the text you can also save that into String

print(pageObj.extractText())
pdfReader.getDocumentInfo()
pdfTxtFile = '../input/cvpr2019/CVPR2019/abstracts/Acuna_Devil_Is_in_the_Edges_Learning_Semantic_Boundaries_From_Noisy_CVPR_2019_paper.txt'

pdf_txt = open(pdfTxtFile, 'r', encoding='utf-8')



# loop over all the lines

for line in pdf_txt:

    print(repr(line))
papers = os.listdir('../input/cvpr2019/CVPR2019/papers/')
data_dict = {'content': [], 'abstract': [], 'authors':[], 'title':[]}
def pdf_to_text(path):

    bashCommand = "pdf2txt.py -o pap.txt " + path

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

    output, error = process.communicate()

    pdfTxtFile = 'pap.txt'

    pdf_txt = open(pdfTxtFile, 'r', encoding='utf-8')

    strn = ''

    # loop over all the lines

    for line in pdf_txt:

        strn += line

    return strn
def read_txt(path):

    abs_text = open(path, 'r', encoding='utf-8')

    strn = ''

    # loop over all the lines

    for line in abs_text:

        strn += line

    return strn
def read_a_paper(name):

    if os.path.exists('../input/cvpr2019/CVPR2019/abstracts/' + name.split('.')[0] + '.txt'):

        data_dict['content'].append(pdf_to_text('../input/cvpr2019/CVPR2019/papers/' + name))

        data_dict['abstract'].append(read_txt('../input/cvpr2019/CVPR2019/abstracts/' + name.split('.')[0] + '.txt'))

        pdfFileObj = open('../input/cvpr2019/CVPR2019/papers/' + name, 'rb')

        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        pdf_meta = pdfReader.getDocumentInfo()

        data_dict['title'].append(pdf_meta['/Title'])

        data_dict['authors'].append(pdf_meta['/Author'])
for paper in tqdm(papers):

    read_a_paper(paper)
len(data_dict['content'])
df = pd.DataFrame(data_dict)
df.to_csv('cvpr2019.csv')