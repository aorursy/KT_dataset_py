# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import Markdown, display
import json
from collections import Counter


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#data set credits : https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv
#biorxiv_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv')
#clean_comm_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv')
#clean_noncomm_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv')
#clean_pmc_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv')

#all_data=pd.concat([biorxiv_data,clean_comm_data,clean_noncomm_data,clean_pmc_data],axis=0).dropna()
#all_data.shape
#all_data.shape[0]
#all_data.head()
filenames_bio = os.listdir('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/')
print("Number of articles retrieved from biorxiv:", len(filenames_bio))
filenames_comm = os.listdir('/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/')
print("Number of articles retrieved from commercial use:", len(filenames_comm))
filenames_custom = os.listdir('/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/')
print("Number of articles retrieved from custom license:", len(filenames_custom))
filenames_noncomm = os.listdir('/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/')
print("Number of articles retrieved from non commercial:", len(filenames_noncomm))

all_files = []

for filename in filenames_bio:
    filename = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/' + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
for filename in filenames_comm:
    filename = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/' + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
for filename in filenames_custom:
    filename = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/' + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
for filename in filenames_noncomm:
    filename = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/' + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
    

file = all_files[100]
print("Dictionary keys:", file.keys())
titles_list=[]
for  file in all_files: 
    for refs in file['bib_entries'].keys(): 
        titles_list.append(file['bib_entries'][refs]["title"]) 
freqs = dict(Counter(titles_list))
freqs_df=pd.DataFrame.from_dict(freqs,orient='index',columns=['freqs'])
freqs_df=freqs_df.sort_values(by='freqs',ascending=False)
freqs_df['title']=freqs_df.index
metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

metadata=metadata.merge(freqs_df,how='left',on='title',suffixes=(False,'_y'))
metadata=metadata.sort_values(by='freqs',ascending=False)
metadata=metadata.dropna(subset=['abstract']) 
metadata.shape
metadata.head()
list(metadata.columns)
def what_do_we_know(match):
    num=0
    for abstract in metadata.iloc[:,8]:
        matched=False
        for terms in match: 
            if terms in abstract.lower() and "covid" in abstract.lower(): matched=True
        if matched:
            if np.isnan(metadata.iloc[num,17]): citations="NaN" 
            else: citations=str(int(metadata.iloc[num,17]))
            display(Markdown('<i> '+metadata.iloc[num,3]+'</i>'+' - '+citations+' citations'))
            sentence3="<ul>"
            for sentence in abstract.split('. '):
                sentence2=sentence.lower()
                matched2=False
                for terms in match: 
                    if terms in sentence2: 
                        matched2=True
                        sentence2=sentence2.replace(terms, '<b>'+terms+'</b>')
                if matched2: 
                 #   display(Markdown("> "+sentence2))
                    sentence3=sentence3+"<li>"+sentence2.replace("\n","")+"</li>"
            #sentence3=sentence3.replace("****","")
            display(Markdown(sentence3+"</ul>"))       
        #    display(Markdown("<sup>"+abstract.replace("\n"," ")+"</sup>"))
        #print(num)
        num+=1
# trial searches
what_do_we_know(["%"," higher than"," lower than"," key result"," equal to"," rate is"," rate was"," p-value"," estimated as"])
#what_do_we_know([" smok"])
#what_do_we_know([" pre-existing"])
#what_do_we_know([" coinfections"," co-infections"," comorbidities"," co-morbidities"])
#what_do_we_know([" neonat"])
#what_do_we_know([" pregn"])
#what_do_we_know([" socioeconomic"," behavioural"])
#what_do_we_know([" economic impact"])
#what_do_we_know([" reproductive number"])
#what_do_we_know([" incubation period"])
#what_do_we_know([" serial interval"])
#what_do_we_know([" transmission"])
#what_do_we_know([" environmental factor"," environment factor"," environment risk"," food"," climate"," sanitation"])
#what_do_we_know([" risk of fatality"," mortality"])
#what_do_we_know([" hospitalized patients"])
#what_do_we_know([" high-risk"," high risk"])
#what_do_we_know([" susceptibility"])
#what_do_we_know([" mitigation measures"," social distan"," mass gathering"," quarantine"," lockdown"," lock-down"," containment"," shutdown"])
