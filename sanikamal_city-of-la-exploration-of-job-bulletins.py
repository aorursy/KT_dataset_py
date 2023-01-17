import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import xml.etree.ElementTree as ET

import zipfile

import os

from os import walk

import shutil

from shutil import copytree, ignore_patterns

from PIL import Image

from wand.image import Image as Img

import matplotlib.pyplot as plt

from collections import Counter

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS



%matplotlib inline
print(os.listdir("../input"))
bulletin_dir = "../input/cityofla/CityofLA/Job Bulletins"

addl_data_dir="../input/cityofla/CityofLA/Additional data"
pdf = '../input/cityofla/CityofLA/Additional data/PDFs/2014/September 2014/09262014/PRINCIPAL INSPECTOR 4226.pdf'

Img(filename=pdf, resolution=200)
pdf = '../input/cityofla/CityofLA/Additional data/PDFs/2018/December/Dec 7/COMMERCIAL FIELD REPRESENTATIVE 1600 120718.pdf'

Img(filename=pdf, resolution=200)
sample_job_template = pd.read_csv(os.path.join(addl_data_dir, 'sample job class export template.csv'))

sample_job_template
data_dictionary = pd.read_csv(os.path.join(addl_data_dir, 'kaggle_data_dictionary.csv'))

data_dictionary.head()


job_titles = pd.read_csv(os.path.join(addl_data_dir, 'job_titles.csv'), names=['JOB TITLES'])

job_titles.head()
job_files = os.listdir(bulletin_dir)

print("No of files in Job Bulletins Folder:",len(job_files))
with open(bulletin_dir+"/"+job_files[0]) as file:

    print("File name: ",file.name)

    print("=====================================")

    print(file.read(1000))
# Code from the starter kernel to iterate over Job Bulletins directory

data_list = []

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        for line in f.readlines():

            #Insert code to parse job bulletins

            if "Class Code:" in line:

                class_code=line.split("Class Code:")[1].split("Open Date")[0].strip()

            if "Open Date:" in line:

                job_bulletin_date = line.split("Open Date:")[1].split("(")[0].strip()

        data_list.append([filename,class_code,job_bulletin_date])
# Form a DataFrame 

df = pd.DataFrame(data_list)

df.columns = ["FILE_NAME","CLASS_CODE","OPEN_DATE"]

df.head()
df.info()
df.OPEN_DATE = pd.to_datetime(df.OPEN_DATE)
df.info()
df.tail()
df.to_csv("job_bulletins.csv",index= False)
job_df = pd.read_csv("job_bulletins.csv")

job_df.head()