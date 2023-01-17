# install libraries

! pip install tabula-py

! apt install ghostscript

! pip install camelot-py[cv]
import pandas as pd

import requests

import PyPDF2

import tabula

import camelot
# load the table data

df_meta = pd.read_csv("../input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv")

print(df_meta.shape)
# save the paper pdf

title = "Clinical Characteristics of Coronavirus Disease 2019 in China"

idx = df_meta[df_meta['title']==title].index[0]

doi = df_meta['doi'][idx]

pdf_path = "https://www.nejm.org/doi/pdf/{}?articleTools=true".format(doi)

print("title: {}".format(title))

print("idx: {}".format(idx))

print("doi: {}".format(doi))

print("pdf_path: {}".format(pdf_path))

r = requests.get(pdf_path)

with open("tmp.pdf",'wb') as file:

        file.write(r.content)
# extract tables using tabula

tables = tabula.read_pdf("tmp.pdf", pages = "all", multiple_tables = True)
# check the shape of extracted tables

for i in range(len(tables)):

    tmp = pd.DataFrame(tables[i])

    print("table {:02d}, shape: {}".format(i+1, tmp.shape))
# check the table 1 at page 5, 6

tables[4].head()
tables[5].head()
# check the table 2 at page 7, 8

tables[6].head()
tables[7].head()
# check the table 3 at page 9, 10

tables[8]
tables[9].head()
# save

tables[4].to_csv("table1_1.csv", index=None)

tables[5].to_csv("table1_2.csv", index=None)

tables[6].to_csv("table2_1.csv", index=None)

tables[7].to_csv("table2_2.csv", index=None)

tables[8].to_csv("table3_1.csv", index=None)

tables[9].to_csv("table3_2.csv", index=None)
# try to extract tables using camelot

tables = camelot.read_pdf('tmp.pdf')
# check the number of extracted tables

print(len(tables))
# delete the pdf

!rm tmp.pdf