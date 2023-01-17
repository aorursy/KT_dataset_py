import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import warnings

warnings.filterwarnings('ignore')
!apt -y install enchant

!pip install pytesseract

!pip install PyDictionary

!pip install googletrans

!pip install pdf2image

!pip install PyPDF2

!pip install fpdf

!pip install pyenchant
import time

import datetime



import requests

import json



from tqdm import tqdm as tq

import pytesseract

import shutil

import os

import random

try:

    from PIL import Image

except ImportError:

    import Image



import re



import pickle

import PyPDF2

from pdf2image import convert_from_path, convert_from_bytes

from pdf2image.exceptions import (

    PDFInfoNotInstalledError,

    PDFPageCountError,

    PDFSyntaxError

)



import nltk

nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from PyDictionary import PyDictionary

nltk.download('stopwords')

nltk.download('words')

stop_words = set(stopwords.words('english'))

english_words = set(nltk.corpus.words.words())



from collections import Counter



from PyDictionary import PyDictionary

dictionary=PyDictionary()

from googletrans import Translator

translator = Translator()



from fpdf import FPDF



import enchant

spellchecker = enchant.Dict("en_US")



base_url = 'https://api.dictionaryapi.dev/api/v1/entries/en/'
# spellchecker.check("hello")
def pdf_to_dic(pdf_dir):

    big_str = ""

    pdfFileObj = open(pdf_dir,'rb')

    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)



    for i in range(pdfReader.numPages):

        pdf_text = pdfReader.getPage(i).extractText()

        big_str += pdf_text

    word_list = re.findall(r"[\w']+", big_str.lower())

    

    short_list = []

    for word in word_list:

        tmp = WordNetLemmatizer().lemmatize(word,'v')

        if tmp not in stop_words and spellchecker.check(tmp) and len(tmp) > 1:

            short_list.append(tmp)        

            

    word_freq = Counter(short_list)

    sorted_dic = sorted(word_freq.items(), key= lambda t: (t[0]), reverse=False)

    

    return sorted_dic
# pdf_dir = "/kaggle/input/pdfbooks/mockingbird.pdf"

# output_file = "mockingbird-dictionary"

# sorted_dic = pdf_to_dic(pdf_dir)

# len(sorted_dic)

#         r = requests.get(url, params=dict(define=con))



# for i in tq(range(100,150)):

#     con = sorted_dic[i][0]

#     try:

#         r = requests.get(base_url+con)

#         if r.status_code == 404:

#             i -= 1

#             print(con)

    #     print(r.json())

    #     if "word" in r.json()[0]:

    #         word = r.json()[0]["word"]

    #         print("word", word)

    #     if "phonetic" in r.json()[0]:

    #         phonetic = r.json()[0]["phonetic"]

    #         print("phonetic", phonetic)

    #     if "origin" in r.json()[0]:

    #         origin = r.json()[0]["origin"]

    #         print("origin", origin)

    #     if "meaning" in r.json()[0]:

    #         meaning = r.json()[0]["meaning"]



    #     for pos in meaning.keys():

    #         print(pos)

    #         for version in meaning[pos]:

    #             for tp in version.keys():

    #                 print(tp, ":\t", version[tp])

    #         print()

    #     print()

#     except:

#         print("exception",con)
# url = 'https://googledictionaryapi.eu-gb.mybluemix.net/'

# r = requests.get(url, params=dict(define='abominable'))

# q = requests.get(base_url+'abominable')

# print(r.json())

# print(q.json())



def create_pdf(sorted_pairs, out_pdf):

    pdf = FPDF() 

    pdf.set_auto_page_break(True, 25)

    pdf.add_page()    

        

    tolerance = 0

    i=0

    while(True):

        if i >= len(sorted_pairs):

            break

        con = sorted_pairs[i][0]

        i += 1

        try:

            r = requests.get(base_url+con)

            if r.status_code == 404:

                tolerance += 1

                if tolerance < 100:

                    i = i - 1

                else:

                    print("not found", con)

                    tolerance = 0          

                continue

            

            if "word" in r.json()[0]:

                word = r.json()[0]["word"]

                pdf.set_text_color(0,0,100)

                tmp = str(word).encode('latin-1', 'replace').decode('latin-1')

                pdf.set_font("Arial", size = 25)

                pdf.multi_cell(200, 10, txt = tmp, align = 'L')

                

            if "phonetic" in r.json()[0]:

                phonetic = r.json()[0]["phonetic"]

                pdf.set_text_color(100,0,0)

                pdf.set_font("Arial", size = 15)

                pdf.multi_cell(200, 10, "phonetic", align = 'L')

                pdf.set_text_color(0,0,0)

                pdf.set_font("Arial", size = 13)

                tmp = str(phonetic).encode('latin-1', 'replace').decode('latin-1')

                pdf.multi_cell(200, 10, tmp, align = 'L')



            if "origin" in r.json()[0]:

                origin = r.json()[0]["origin"]

                pdf.set_text_color(100,0,0)

                pdf.set_font("Arial", size = 15)

                pdf.multi_cell(200, 10, "origin", align = 'L')

                pdf.set_text_color(0,0,0)

                pdf.set_font("Arial", size = 13)

                tmp = str(origin).encode('latin-1', 'replace').decode('latin-1')

                pdf.multi_cell(200, 10, tmp, align = 'L')



            if "meaning" in r.json()[0]:

                meaning = r.json()[0]["meaning"]

                        

                pdf.set_text_color(0,100,0)

                pdf.set_font("Arial", size = 18)

                pdf.multi_cell(200, 10, "meaning", align = 'L')



                for pos in meaning.keys():

                    pdf.set_text_color(100,0,100)

                    tmp = str(pos).encode('latin-1', 'replace').decode('latin-1')

                    pdf.set_font("Arial", size = 15)

                    pdf.multi_cell(200, 10, tmp, align = 'L')



                    pdf.set_text_color(0,0,0)

                    pdf.set_font("Arial", size = 13)



                    for version in meaning[pos]:

                        for tp in version.keys():

                            tmp = (str(tp) + ":\t" + str(version[tp])).encode('latin-1', 'replace').decode('latin-1')

                            pdf.multi_cell(200, 10, tmp, align = 'L')

                        pdf.multi_cell(200, 10, "\n", align = 'L')

                pdf.multi_cell(200, 10, "\n", align = 'L')

        except:

            print("exception",con)

        

    pdf.output(out_pdf)
def prepare_pdf(pdf_dir,out_pdf):

    out_pdf = out_pdf + ".pdf"

    sorted_dic = pdf_to_dic(pdf_dir)

    create_pdf(sorted_dic, out_pdf)
pdf_dir = "/kaggle/input/pdfbooks/mockingbird.pdf"

output_file = "mockingbird-dictionary"



start = time.time()

prepare_pdf(pdf_dir,output_file)

end = time.time()

sec = end - start

print(datetime.timedelta(seconds=sec))
!ls
# import webbrowser

# webbrowser.open(r'mockingbird-dictionary.pdf')