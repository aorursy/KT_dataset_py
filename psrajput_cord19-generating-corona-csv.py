import numpy as np, pandas as pd, os, json, re
from tqdm.notebook import tqdm
BM_pdf_loaded, CUS_pmc_loaded, CUS_pdf_loaded, CL_pmc_loaded, CL_pdf_loaded, NUS_pmc_loaded, NUS_pdf_loaded = [], [], [], [], [], [], []

BM_pdf = "/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/"

CUS_pmc = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pmc_json/"
CUS_pdf = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/"

CL_pmc = "/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pmc_json/"
CL_pdf = "/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/"

NUS_pmc = "/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pmc_json/"
NUS_pdf = "/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/"

def jsonAppender(dir_, loaded_file):
    
    files = os.listdir(dir_)
    
    for filename in tqdm(files):
        file = dir_ + filename
        opened_file = json.load(open(file, 'rb'))
        loaded_file.append(opened_file)
        
jsonAppender(BM_pdf, BM_pdf_loaded)
jsonAppender(CUS_pmc, CUS_pmc_loaded)
jsonAppender(CUS_pdf, CUS_pdf_loaded)
jsonAppender(CL_pmc, CL_pmc_loaded)
jsonAppender(CL_pdf, CL_pdf_loaded)
jsonAppender(NUS_pmc, NUS_pmc_loaded)
jsonAppender(NUS_pdf, NUS_pdf_loaded)


print(
    "BM_pdf Count", len(BM_pdf_loaded), 
    "\nCUS_pmc Count", len(CUS_pmc_loaded), 
    "\nCUS_pdf Count", len(CUS_pdf_loaded), 
    "\nCL_pmc Count", len(CL_pmc_loaded), 
    "\nCL_pdf Count", len(CL_pdf_loaded), 
    "\nNUS_pmc Count", len(NUS_pmc_loaded), 
    "\nNUS_pdf Count", len(NUS_pdf_loaded),
)
# CLeaning Helpers

def authorFormator(file):
    
    Authors = ""
    AuthorsO = []
    
    for i in file["metadata"]["authors"]:

        firstName = str(i["first"])
        middleName = str(i["middle"])
        lastName = str(i["last"])
        suffix = str(i["last"])

        if (middleName != "[]"):
            middleName = middleName.replace("['","").replace("']","")
            Author = firstName + " " + middleName + " " + lastName
        else:
            Author = firstName + " " + lastName
        
        AuthorsO.append(Author)
        
    Authors = ', '.join(AuthorsO)
        
    return Authors

def bodyText(file):
    
    Body_text = ""
    for i in file["body_text"]:
        Body_text += (i["text"] + "\n\n")
        
    return Body_text
def dataAppender(fileName, clean_file_name):
    for file in tqdm(fileName):

        features = [
            file["paper_id"],
            file["metadata"]["title"],
            authorFormator(file),
            bodyText(file)

        ]
        
        clean_file_name.append(features)
# Cleaning BM_pdf

BM_pdf_cleaned = []
dataAppender(BM_pdf_loaded, BM_pdf_cleaned)
# Cleaning CUS_pmc

CUS_pmc_cleaned = []
dataAppender(CUS_pmc_loaded, CUS_pmc_cleaned)
# Cleaning CUS_pdf

CUS_pdf_cleaned = []
dataAppender(CUS_pdf_loaded, CUS_pdf_cleaned)
# Cleaning CL_pmc

CL_pmc_cleaned = []
dataAppender(CL_pmc_loaded, CL_pmc_cleaned)
# Cleaning CL_pdf

CL_pdf_cleaned = []
dataAppender(CL_pdf_loaded, CL_pdf_cleaned)
# Cleaning NUS_pmc

NUS_pmc_cleaned = []
dataAppender(NUS_pmc_loaded, NUS_pmc_cleaned)
# Cleaning NUS_pdf

NUS_pdf_cleaned = []
dataAppender(NUS_pdf_loaded, NUS_pdf_cleaned)
cols = [
    'Paper_id', 
    'Title', 
    'Authors',
    'Body_text'
]

BM_pdf_df = pd.DataFrame(BM_pdf_cleaned, columns=cols)
CUS_pmc_df = pd.DataFrame(CUS_pmc_cleaned, columns=cols)
CUS_pdf_df = pd.DataFrame(CUS_pdf_cleaned, columns=cols)
CL_pmc_df = pd.DataFrame(CL_pmc_cleaned, columns=cols)
CL_pdf_df = pd.DataFrame(CL_pdf_cleaned, columns=cols)
NUS_pmc_df = pd.DataFrame(NUS_pmc_cleaned, columns=cols)
NUS_pdf_df = pd.DataFrame(NUS_pdf_cleaned, columns=cols)

dfs = [BM_pdf_df, CUS_pmc_df, CUS_pdf_df, CL_pmc_df, CL_pdf_df, NUS_pmc_df, NUS_pdf_df]

clean_df = pd.concat(dfs)

clean_df.head()
"""
Clean Corona CSV file
Created on 8 April 2020
File Shape: (52097, 4)
"""

# clean_df.to_csv('cleanCORD.csv', index=False)
import pandas as pd
data = pd.read_csv("../input/cleancord/cleanCorona.csv")
data.head()
# Preparing covid_keywords
covid_keywords_list = [" nCov", 
                        "COVID-19",
                        "COVID 19",
                        "2019 nCov",
                        "2019-nCov", 
                        "COVID 2019", 
                        "SARS-CoV-2", 
                        "SARS CoV-2", 
                        "SARS CoV 2", 
                        "Coronavirus 2", 
                        "Orthocoronavirinae",
                        "SARS Coronavirus 2",
                        "2019 Novel Coronavirus", 
                        "2019 Coronavirus Pandemic",
                        "Coronavirus Disease 2019", 
                        "2019-nCoV acute respiratory disease", 
                        "Novel coronavirus pneumonia"
                      ]
covid_keywords = '|'.join(covid_keywords_list)
# Filtering COVID data
covid_19_bool = data.Body_text.str.contains(covid_keywords, na=False, case=False)
# Generating COVID data

covid_19 = data[covid_19_bool]
covid_19.shape
covid_19.head()
# Replacing Unnecessary Characters with space
covid_19_f = covid_19.replace({'<br>|\n|► ': ' '}, regex=True)
covid_19_f.head()
"""
Exporting COVID CSV
Available On: https://www.kaggle.com/psrajput/covid-19
Last Updated: 9 April
"""

# covid_19_f.to_csv(r'covid_19.csv')