# Standard imports and imports for creating a .csv
import numpy as np
import os
import csv
import urllib
import json 
import zipfile
import pandas as pd

!pip install PyPDF2
!pip install --upgrade pip
!pip install requests



# Import libraries for parsing
import PyPDF2 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import fnmatch

def get_files_words():
    path = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/PDFs'
    filesread = 0
        
    filesread = 0
    files = [os.path.join(dirpath, filename)
        for dirpath, dirnames, files in os.walk(path)
        for filename in files if filename.endswith('.pdf')]

    filecount=len(files)
        
    while filesread <  filecount:
            for file in files:
                    csv_file_name = 'job_descriptions.csv'

                    headers = ['Job Description', "Year"]
                    
                    fileerror = 0
                    filegood = 0
                      
                    if os.path.isfile(csv_file_name):
                            with open(csv_file_name, 'a') as outfile:
                                    writer = csv.writer(outfile)
                                    namesonly = file.split("/")
                                    arraylen=len(namesonly)
                                    try:
                                        fileonly = namesonly[arraylen-1]
                                        fileyear = namesonly[8]
                                        writer.writerow([fileonly, fileyear])
                                        filegood +=1
                                    except:
                                        fileerror +=1
                    else:
                            with open(csv_file_name, 'w') as outfile:
                                    writer = csv.writer(outfile)
                                    writer.writerow(headers)
                                    namesonly = file.split("/")
                                    arraylen=len(namesonly)
                        
                                    try:
                                        fileonly = namesonly[arraylen-1]
                                        fileyear = namesonly[8]
                                        writer.writerow([fileonly, fileyear])
                                        filegood +=1
                                    except:
                                        fileerror +=1
                                        
                    filesread +=1
 
                    pdfFileObj = open(file,'rb')               ## open the file
                    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)   ## read the data in the file
                    num_pages = pdfReader.numPages                 ## figure out the number of pages
                    count = 0
                    text = ""

                    while count < num_pages:                       ## get a page
                        pageObj = pdfReader.getPage(count)          
                        count +=1
                        text += pageObj.extractText()              ## extract the data from the page
                        if text != "":
                            text = text
                                    
                        else:
                            text = textract.process(fileurl, method='tesseract', language='eng')
                            
                    global tokens
                    tokens = word_tokenize(text)
                    textlen = len(text) 
                    
            print (filesread, "files read")
            print ("words in file: ",textlen)

             


get_files_words()
punctuations = ['(',')',';',':','[',']',',', "$", "%", "/"]
web_characters =[ "http//", "http", ".", ['.'], "-", "www", "//" "https"]
stop_words = stopwords.words('english')
def get_key_words():  
    
    keywords = [word for word in tokens if not word in stop_words \
                and not word in punctuations and not word in web_characters]

    keywordct = len(keywords)
    print("keyword count", keywordct)


    unique_keywords = set(keywords) 
    uniquect = len(unique_keywords)
    print("unique count", uniquect)


    wordread = 0
    global goodwords
    goodwords=[]

    while wordread < keywordct:
            wordread +=1
            
    goodwords = [word.lower() for word in unique_keywords if word.isalpha()]

    global goodwordct
    goodwordct = len(goodwords)
    print("good words", goodwordct) 

get_key_words()
import requests

index = 0
wordcode = 0 
# goodwordct=10
dictlist = []
dropword = 0
twinword_key = "null"

while wordcode < goodwordct:
    myword = goodwords[index]
    wordcode +=1
    index +=1    

    response = requests.get("https://twinword-language-scoring.p.rapidapi.com/word/",
            headers={
                "X-RapidAPI-Host": "twinword-language-scoring.p.rapidapi.com",
                "X-RapidAPI-Key": twinword_key,
                "Content-Type": "application/x-www-form-urlencoded"
                       },
                params={
                  "entry": myword
                }
        )
    
    if response.status_code == 200:
        try:
            data =response.json()
            worddiff = data.get("ten_degree")
            if type(worddiff) == int:
                listdata = (worddiff, myword)
                dictlist.append(listdata)
            else:
                print ("Word Difficulty not found for: ", myword)
        except:
            dropword +=1
        
print ("words dropped", dropword)
def write_to_csv():
    dict_list_sorted = sorted(dictlist, key=None, reverse=True)
    output_dataframe = pd.DataFrame(dict_list_sorted,columns=["Word Difficulty",'Word']) 
    csv_outfile_name = 'word_difficulty.csv'
    output_dataframe.to_csv(csv_outfile_name , index=False)
    
write_to_csv()
Male_Bias_csv = "../input/bais-word-lists/Female-Bias-wordparts.csv"
with open(Male_Bias_csv, 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    male_bias_words = list(reader)
Female_Bias_csv = "../input/bais-word-lists/Female-Bias-wordparts.csv"
with open(Female_Bias_csv, 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    female_bias_words = list(reader)

dict_list_sorted = sorted(dictlist, key=None, reverse=True)
gender_clean_dataframe = pd.DataFrame(dict_list_sorted)
gender_clean_dataframe.columns = ["Difficulty","Word"]
gender_df =  gender_clean_dataframe["Word"]
gender_list = gender_df.values.tolist()
male_matchbias = []
try:
    male_matchbias = [s for s in gender_list if any(xs in s for xs in male_bias_words)]
except:
    pass

if (len(male_matchbias)) == 0:
    print ("No male bias words were found")
else:
    print("Male bias words were found and exported")
    male_bias_out= pd.DataFrame(male_matchbias)
    male_bias_out.columns = ["Male Bias Words"]
    csv_outfile_name = 'Male Bias Words.csv'
    male_bias_out.to_csv(csv_outfile_name , index=False, header =False)
female_matchbias = []
try:
    female_matchbias = [s for s in gender_list if any(xs in s for xs in female_bias_words)]
except:
    pass


if (len(female_matchbias)) == 0:
    print ("No female bias words were found")
else:
    print("female bias words were found and exported")
    female_bias_out= pd.DataFrame(female_matchbias)
    female_bias_out.columns = ["female Bias Words"]
    csv_outfile_name = 'female Bias Words.csv'
    female_bias_out.to_csv(csv_outfile_name , index=False, header =False)

Age_Bias_csv = "../input/bais-word-lists/age-bias-wordparts.csv"
with open(Age_Bias_csv, 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    age_bias_words = list(reader)
age_matchbias = []
try:
    age_matchbias = [s for s in gender_list if any(xs in s for xs in age_bias_words)]
except:
    pass


if (len(age_matchbias)) == 0:
    print ("No age bias words were found")
else:
    print("Age bias words were found and exported")
    age_bias_out= pd.DataFrame(age_matchbias)
    age_bias_out.columns = ["Age Bias Words"]
    csv_outfile_name = 'Age Bias Words.csv'
    female_bias_out.to_csv(csv_outfile_name , index=False, header =False)

