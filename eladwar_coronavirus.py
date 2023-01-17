!pip install PyPDF2
import numpy as np 

import pandas as pd



import PyPDF2



import os
files =  []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))
files = pd.Series(files)
pmc_sources = []

comm_use_sources = []

biorxiv_medrxiv_sources = []

noncomm_use_subset_sources = []

for source in list(files.values):

    if ('pmc_custom_license' in source):

        pmc_sources.append(source)

    elif ('noncomm_use_subset' in source):

        noncomm_use_subset_sources.append(source)   

    elif ('comm_use_subset' in source):

        comm_use_sources.append(source)

    elif ('biorxiv_medrxiv' in source):

        biorxiv_medrxiv_sources.append(source)
print(np.array(pmc_sources).shape, np.array(comm_use_sources).shape, np.array(biorxiv_medrxiv_sources).shape, np.array(noncomm_use_subset_sources).shape)
# creating an object 

file = open('/kaggle/input/CORD-19-research-challenge/2020-03-13/COVID.DATA.LIC.AGMT.pdf', 'rb')



# creating a pdf reader object

fileReader = PyPDF2.PdfFileReader(file)

pageObj = fileReader.getPage(0) 

print(pageObj)

print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')

print(pageObj.extractText()) 
import json

with open('/kaggle/input/CORD-19-research-challenge/2020-03-13/json_schema.txt','r', encoding='utf-8') as f:

    print(f.read())
All_Sources_Metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
All_Sources_Metadata
All_Sources_Metadata.columns
All_Sources_Metadata.shape
Key_Metadata = All_Sources_Metadata[['title','abstract','journal','publish_time']]