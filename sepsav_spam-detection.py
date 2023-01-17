# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from nltk.tokenize import sent_tokenize, word_tokenize

data = "Butuh dana cepat"

print(word_tokenize(data))
keywords = ['gadai','butuh','modal','usaha','dana','cepat','tenor','curhat','juta','bpkb','stnk','sedot','wc']
def count_keyword(text,keywords):

    text = text.lower()

    count = 0

    for key in keywords:

        if(key in text):

            count += 1

    return count
count_keyword(data,keywords)
def count_number(text):

    count = 0

    for i in text:

        if(i.isdigit()):

            count+=1

    return count
count_number('hubungi ria 08299')
import re

#perlu di adjust lagi

def contain_link(text):

    for word in text.split(" "):

        if('/' in word and '.' in word):

            return True

    return False

                     
contain_link('haha yeye http://103.10.200.61')
def is_spam(text,keywords):

    if (count_keyword(text,keywords)>1 or count_number(text)>5 or contain_link(text)):

        return True

    return False
text1 = 'Info dari SHOPEE INDONESIA Anda M-dpat CEK Rp.125,jt PIN:(AAQ2099) U/info hadiah klik di www.hadiah-kejutan99.blogspot.com'

text2 = 'WOM F.I.N.A.N.C.E AGUNAN MOTOR/MOBIL Pencairan 50Jt-5M Proses Cepat Dan Mudah 30 MENIT CAIR RIA 082110034422Call/Wa Kami Terbesar Dan Terpercaya'

text3 = 'No.Anda Mendapat HADIAH Cek 100jt dari TELK0MSEL Kode PIN Pemenang Anda (ijh76k79) U/Info klik; www.berkahisiulang-2018.blogspot.com'
text_1 = 'Salam satu hati selamat siang kami dari bengkel ingin konfirmasi motornya sudah selesai dan bisa di ambil. Kami tutup jam 5. Terimakasih'
print(is_spam(text1,keywords))

print(is_spam(text2,keywords))

print(is_spam(text3,keywords))
print(is_spam(text_1,keywords))

#print(is_spam(text2,keywords))

#print(is_spam(text3,keywords))
from urllib.parse import urlparse



a = 'http://www.cwi.nl:80/%7Eguido/Python.html'

b = 'http://bit.ly/2rvobuJ'

c = 'goo.gl/JNFhmW'

d = u'dkakasdkjdjakdjadjfalskdjfalk'



def uri_validator(x):

    try:

        result = urlparse(x)

        return all([result.scheme, result.netloc, result.path])

    except:

        return False

print(uri_validator(a))

print(uri_validator(b))

print(uri_validator(c))

print(uri_validator(d))