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
# Converting pubmed_id, Microsoft academic Paper ID, doi to str as by default it is coming as float.
# converting has_pdf_parse, has_pmc_xml_parse to bool type
meta = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv',dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str,
    'has_pdf_parse': bool,
    'has_pmc_xml_parse': bool
})
meta.head()
# No of columns
len(meta.columns)
# Checking the dataset and their reccords against the cord_uid
meta.count()
# checking different values for license and values associated with it.
meta.groupby("license")["cord_uid"].count()
# checking different values for "has_pdf_parse" and values associated with it.
print(meta.groupby("has_pdf_parse")["cord_uid"].count())
print(meta.groupby("has_pmc_xml_parse")["cord_uid"].count())

# checking the condition when both of them are true.
print(meta[(meta["has_pdf_parse"] == False) & (meta["has_pmc_xml_parse"] == True)].count())
# current directory
import os
os.getcwdb()
# checking number of the json files by checking files inside
import glob
kaggle_cord_data_root_path = "../input/CORD-19-research-challenge"
articles_path = glob.glob('{}/**/*.json'.format(kaggle_cord_data_root_path), recursive=True)
len(set((articles_path)))
