import os
import numpy as np 
import pandas as pd 
import json
from tqdm.notebook import tqdm
NUS_pmc = "noncomm_use_subset/pmc_json/"
NUS_pdf = "noncomm_use_subset/pdf_json/"
NUS_pdf_loaded = []
NUS_pmc_loaded = []

def jsonAppender(dir_, loaded_file):
    files = os.listdir(dir_)
    for filename in tqdm(files):
        file = dir_ + filename
        opened_file = json.load(open(file, 'rb'))
        loaded_file.append(opened_file)

jsonAppender(NUS_pdf, NUS_pdf_loaded)
jsonAppender(NUS_pmc, NUS_pmc_loaded)
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

def bodySection(file):
    
    Body_Section = ""
    section = ""
    for i in file["body_text"]:
        if Body_Section != i["section"]:
            Body_Section += " " + (i["section"] )
            section += Body_Section + ","
    return section

def affiliation(file):
    institutions = ""
    institution0 = []
    
    for i in file["metadata"]["authors"]:
        if 'institution' in i["affiliation"]:
            institution = (i["affiliation"]["institution"])
            institution0.append(institution)
    institutions = ', '.join(institution0)
#     print(institutions)
    return institutions

def dataAppender(fileName, clean_file_name):
    for file in tqdm(fileName):
        features = [
            file["paper_id"],
            file["metadata"]["title"],
            authorFormator(file),
            bodyText(file),
            bodySection(file),
            affiliation(file)
        ]
        clean_file_name.append(features)
NUS_pdf_cleaned = []
dataAppender(NUS_pdf_loaded, NUS_pdf_cleaned)
# NUS_pdf_cleaned[0]
NUS_pmc_cleaned = []
dataAppender(NUS_pmc_loaded, NUS_pmc_cleaned)
cols = [
    'Paper_id', 
    'Title', 
    'Authors',
    'Body_text',
    'Body_Section',
    'institutions'
]
NUS_pmc_df = pd.DataFrame(NUS_pmc_cleaned, columns=cols)
NUS_pdf_df = pd.DataFrame(NUS_pdf_cleaned, columns=cols)

dfs = [NUS_pmc_df, NUS_pdf_df]

clean_df = pd.concat(dfs)
clean_df["institutions"].head()
clean_df.to_csv('cleanCORD.csv', index=False)
