import numpy as np

import pandas as pd

import os

import xml.etree.ElementTree as ET

import zipfile



addl_dir = "../input/cityofla/CityofLA/Additional data"



# First up: An example of what LA would like our output to look like at the end of the day

sample_template = pd.read_csv(os.path.join(addl_dir, 'sample job class export template.csv'))

display(sample_template)
# The data dictionary for reference:

data_dictionary = pd.read_csv(os.path.join(addl_dir, 'kaggle_data_dictionary.csv'))

display(data_dictionary)
# Job titles listing:

job_titles = pd.read_csv(os.path.join(addl_dir, 'job_titles.csv'), names=['JOB TITLES'])

display(job_titles.head())
class Docx(object):

    '''Helper to pull Docx text into a python notebook.

    

    Args:

        - path (str): where the docx file is stored.

    Methods:

        - read: open the doc, return its text.

    '''

    

    def __init__(self, path):

        self.path = path



    def read(self):

        '''Open the doc, return its text.'''

        zipf = zipfile.ZipFile(self.path)

        filelist = zipf.namelist()

        text = Docx.xml2text(zipf.read('word/document.xml'))

        zipf.close()

        return text



    @staticmethod

    def qn(tag):

        '''Source: docx2txt'''

        nsmap = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

        prefix, tagroot = tag.split(':')

        uri = nsmap[prefix]

        return '{{{}}}{}'.format(uri, tagroot)



    @staticmethod

    def xml2text(xml, mapping=None, version='new'):

        '''Source: docx2txt'''

        if mapping is None:

            if version == 'default':

                mapping = {'tab': '\t',

                           'break': '\n',

                           'page': '\n\n'}

            elif version == 'new':

                mapping = {'tab':' ', 'break':' ', 'page':' '}

        text = u''

        root = ET.fromstring(xml)

        for child in root.iter():

            if child.tag == Docx.qn('w:t'):

                t_text = child.text

                text += t_text if t_text is not None else ''

            elif child.tag == Docx.qn('w:tab'):

                text += mapping['tab']

            elif child.tag in (Docx.qn('w:br'), Docx.qn('w:cr')):

                text += mapping['break']

            elif child.tag == Docx.qn("w:p"):

                text += mapping['page']

        return text
print('`Description of promotions in job bulletins`.docx contents:\n')

print(Docx(os.path.join(addl_dir, 'Description of promotions in job bulletins.docx')).read())
!apt-get install -y poppler-utils

# Note: Internet must be enabled on this kernel (under settings on right-hand side)
import tempfile

import shlex

from subprocess import Popen, PIPE, STDOUT



class Pdf(object):

    '''Parse *Searchable* PDF objects.

    Args:

        - path (str): where the docx file is stored.

    Methods:

        - read: open the doc, return its text.

    '''

    def __init__(self, path):

        self.path = path



    def read(self):

        local_filename = tempfile.NamedTemporaryFile(delete=False).name

        run_command_line(f'pdftotext {self.path} {local_filename} -layout')

        with open(local_filename, 'r') as f:

            return '\n'.join(f.readlines())



def run_command_line(command_line_args):

    '''Utility for running command line program via python

    Args:

        - command_line_args (str): what should be passed to command line

    Returns:

        - Bool: (not process.wait()) == 1 if program exited without error

    '''

    process = Popen(shlex.split(command_line_args), stdout=PIPE, stderr=STDOUT)

    with process.stdout:

        return parse_subprocess(process.stdout)

    return not process.wait()



def parse_subprocess(pipe):

    '''Parse output from subprocess.'''

    return [line for line in iter(pipe.readline, b'')]
# It doesn't seem like any of these PDFs are searchable, would need to turn to OCR for parsing.

Pdf(os.path.join(addl_dir, 'PDFs/2014/September 2014/09262014/CHIEF INSPECTOR 4254.pdf')).read()
# Code from the starter kernel to iterate over Job Bulletins directory

bulletin_dir = "../input/cityofla/CityofLA/Job Bulletins"

data_list = []

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        raw = []

        for line in f.readlines():

            raw.append(line)

            if "Open Date:" in line:

                job_bulletin_date = line.split("Open Date:")[1].split("(")[0].strip()

        data_list.append([filename, job_bulletin_date, raw])



# Form a DF 

df = pd.DataFrame(data_list, columns=["FILE_NAME", "OPEN_DATE", "RAW"])

df.OPEN_DATE = pd.to_datetime(df.OPEN_DATE)

df.head()
# Example listing for viewing

# TBD: parsing of this data (!)

df.RAW[0]
# Very rough estimate of the number of words via base python

df.loc[:,'NUM_WORDS'] = [' '.join(lines).split(' ') for lines in df.loc[:,'RAW']]

df.loc[:,'NUM_WORDS'] = [len([word for word in words if word!='' and word!='\n']) for words in df.loc[:,'NUM_WORDS']]
df.NUM_WORDS.describe()
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111)

ax.hist(df.loc[:,'NUM_WORDS'], edgecolor='black')

ax.set_xlabel('Number of words in job posting')

_=ax.set_ylabel('Number of occurrences')