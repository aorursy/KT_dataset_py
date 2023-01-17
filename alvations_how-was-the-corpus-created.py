!pip install -U python-docx

!pip install -U lazyme
import re

import requests

from bs4 import BeautifulSoup

from docx import Document



from lazyme import find_files



headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}



def save_binary(url, filename, chunk_size=1024*1024*2):

    response = requests.get(url, headers=headers, stream=True)

    with open(filename, 'wb') as fout:

        for chunk in response.iter_content(chunk_size):

            fout.write(chunk)
url = "https://www.microsoft.com/en-us/translator/education/parent-teacher-conference-letters/"

response = requests.get(url, headers=headers)
pattern = r".docx"

bsoup = BeautifulSoup(response.content.decode('utf8'), 'lxml')

docx_links = [link.attrs["href"] for link in 

              bsoup.find_all('a', attrs={"href": re.compile(pattern)})]
import os



os.makedirs('ms-edu-letters')

os.makedirs('ms-ptcl-corpus')
for url in docx_links:

    filename = 'ms-edu-letters/' + url.split('/')[-1]

    save_binary(url, filename)
lang2paragraphs = {}

for filename in sorted(find_files('ms-edu-letters/', '*.docx')):

    if "~" in filename:

        continue

    doc = Document(filename)

    lang = '-'.join(filename.split('/')[-1].split('-')[4:]).split('.')[0]

    lang2paragraphs[lang] = [para.text for para in doc.paragraphs if para.text.strip()]
# Persian correction.

lang2paragraphs['Persian'] = lang2paragraphs['Persian'][:-5] + [' '.join(lang2paragraphs['Persian'][-5:-3])] + lang2paragraphs['Persian'][-3:]

lang2paragraphs['Welsh'] = lang2paragraphs['Welsh'][:-5] + [' '.join(lang2paragraphs['Welsh'][-5:-3])] + lang2paragraphs['Welsh'][-3:]



# Yucatec correction

lang2paragraphs['Yucatec-Maya'] = lang2paragraphs['Yucatec-Maya'][:2] + [' '.join(lang2paragraphs['Yucatec-Maya'][2:4])] + lang2paragraphs['Yucatec-Maya'][4:]

lang2paragraphs['Yucatec-Maya'] = lang2paragraphs['Yucatec-Maya'][:-6] + [' '.join(lang2paragraphs['Yucatec-Maya'][-6:-4])] + lang2paragraphs['Yucatec-Maya'][-4:]

for lang in lang2paragraphs:

    with open(f'ms-ptcl-corpus/ms-ptcl.{lang}.txt', 'w') as fout:

        fout.write('\n'.join(lang2paragraphs[lang]))