# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import datetime as dt

from datetime import timedelta

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

from sklearn.metrics import mean_squared_error,r2_score

import statsmodels.api as sm

from fbprophet import Prophet

covid=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

#covid.head()
covid_india=covid[covid['Country/Region']=="India"]

covid_usa=covid[covid['Country/Region']=="US"]



#Converting the date into Datetime format

covid_india["ObservationDate"]=pd.to_datetime(covid_india["ObservationDate"])

covid_usa["ObservationDate"]=pd.to_datetime(covid_usa["ObservationDate"])

#Taking the data starting 1st March 2020 as that is most relevant for India

start_date = pd.to_datetime("1-Mar-2020")

covid_india_MarchOnwards = covid_india[covid_india["ObservationDate"] > start_date]

covid_usa_MarchOnwards = covid_usa[covid_usa["ObservationDate"] > start_date]



#print(covid_india_MarchOnwards)

covid_india_MarchOnwards.head()

covid_usa_MarchOnwards.head()

#Grouping the data based on the Date 

aggByDate_India=covid_india_MarchOnwards.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

aggByDate_USA=covid_usa_MarchOnwards.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

print("Number of Confirmed Cases India",aggByDate_India["Confirmed"].iloc[-1])

print("Number of Recovered Cases",aggByDate_India["Recovered"].iloc[-1])

print("Number of Death Cases",aggByDate_India["Deaths"].iloc[-1])

print("Number of Active Cases",aggByDate_India["Confirmed"].iloc[-1]-aggByDate_India["Recovered"].iloc[-1]-aggByDate_India["Deaths"].iloc[-1])

print("Number of Closed Cases",aggByDate_India["Recovered"].iloc[-1]+aggByDate_India["Deaths"].iloc[-1])

print("Approximate Number of Confirmed Cases per day",round(aggByDate_India["Confirmed"].iloc[-1]/aggByDate_India.shape[0]))

print("Approximate Number of Recovered Cases per day",round(aggByDate_India["Recovered"].iloc[-1]/aggByDate_India.shape[0]))

print("Approximate Number of Death Cases per day",round(aggByDate_India["Deaths"].iloc[-1]/aggByDate_India.shape[0]))

print("Number of New Cofirmed Cases in last 24 hours are",aggByDate_India["Confirmed"].iloc[-1]-aggByDate_India["Confirmed"].iloc[-2])

print("Number of New Recoverd Cases in last 24 hours are",aggByDate_India["Recovered"].iloc[-1]-aggByDate_India["Recovered"].iloc[-2])

print("Number of New Death Cases in last 24 hours are",aggByDate_India["Deaths"].iloc[-1]-aggByDate_India["Deaths"].iloc[-2])

plt.figure(figsize=(10,5))

#plt.plot(aggByDate_India["Confirmed"],label="Confirmed Cases India")

#plt.plot(aggByDate_India["Recovered"],label="Recovered Cases India")

plt.plot(aggByDate_India["Deaths"],label="Death Cases India")

#plt.plot(aggByDate_USA["Confirmed"],label="Confirmed Cases US")

#plt.plot(aggByDate_USA["Recovered"],label="Recovered Cases US")

plt.plot(aggByDate_USA["Deaths"],label="Death Cases US")



plt.xticks(rotation=90)

plt.ylabel("Number of Cases: Log Scale")

plt.xlabel("Date")

plt.yscale("log")

plt.title("COVID Fatalities Trends in India and USA")

plt.legend()
# Let's explore CORD-19 data. To begin with let's find all those antivirals that have been mentioned in 

# context of India along with reference count

# To start with, let us focus only on PDFs where text has been successfully extracted

# Per https://arxiv.org/pdf/2004.10706.pdf, Each PDF parse has an associated SHA 

# while each XML parse is named using its associated PMC ID.

cord19mdata=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

#Since we are starting with PDFs only - which has about 80% of articles, remove entries that dont have SHA

cord19mdata=cord19mdata[cord19mdata["sha"]==cord19mdata["sha"]] #filters out entries having sha=NaN

#Next find those PDFs where text extraction was successful

#cord19mdata=cord19mdata[cord19mdata["has_pdf_parse"]] #filters out entries for which pdf parse is not available

shaTojsonPath = {}

#since sha entry is a pointer to json file, lets have a method that can give full path to json file



allFileNamestoPath ={}

for dirname, _, files in os.walk('/kaggle/input/CORD-19-research-challenge'):

    for filename in files:

        allFileNamestoPath[filename] = os.path.join(dirname,filename)



#def shatoJSONFilePath(shaid): #returns path of .json file

    #for dirname, _, files in os.walk('/kaggle/input/CORD-19-research-challenge'):

        #if shaid+'.json' in files:

            #return os.path.join(dirname,shaid+".json")

def shatoJSONFilePath(shaid):

    return allFileNamestoPath.get(shaid+".json")



for shaid in cord19mdata["sha"]:

    shaTojsonPath[shaid] = shatoJSONFilePath(shaid)

#Since we would need json file for most of our analysis, let's add json file path to metadata itself

#cord19mdata["jsonPath"]=cord19mdata.apply(lambda x: shatoJSONFilePath(x["sha"]),axis=1) 

#remove rows where we could not find jsonPath

#cord19mdata=cord19mdata[cord19mdata["jsonPath"]==cord19mdata["jsonPath"]]

#cord19mdata.shape
import json

import re

#Now open each json file as per jsonPath (technically, that means do json.load on file at jsonPath)

#There might be multiple body_text nodes in json file, aggregate all body_text commentary as the text in PDF 

#(technically, start with empty '' pdfText and keep doing pdfText = pdfText + body_text_entry)

#Create a mapping of shaId with fullBodyText (technically Python dictionary shaToPDFText)



shaToPDFText = {}

#for shaid,jsonPath in zip(cord19mdata["sha"],cord19mdata["jsonPath"]):

for shaid in cord19mdata["sha"]:

    jsonPath = shaTojsonPath.get(shaid)

    pdfText = ''

    if (jsonPath is not None):

        with open(jsonPath, 'r') as jsonfile:

            jsonfileObj = json.load(jsonfile)

            for body_text_entry in jsonfileObj["body_text"]:

                pdfText = pdfText + (re.sub('[^a-zA-Z0-9]', ' ', body_text_entry["text"].lower()))

    #print(pdfText)

    shaToPDFText[shaid] = pdfText

#Now let's find out what drugs are being talked about

#We want to match all stems and affixes from https://en.wikipedia.org/wiki/Drug_nomenclature



drugStemsAffixes = ['vir ', 'cillin ', ' cef', 'mab ', 'ximab ', 'zumab ', 'ciclib ', 'lisib ', 'tinib ', 'vastatin ', 'prazole ', 'lukast ', 'grel','axine ','olol ','oxetine ','sartan ','pril ','oxacin ','uine ','barb','xaban ','afil ', 'prost','quine ','parib ', 'tide ', 'vec ', 'imsu ','-caine ','sone ']

drugsInPDFs = []

cytoDrugsInPDFs = []

MIN_DRUG_LEN = 6

def singleStranded(pdfText):

    return ((pdfText.rfind('single-stranded')>0) or (pdfText.rfind('single stranded')>0))

def positiveSense(pdfText):

    return ((pdfText.rfind('positive-sense')>0) or (pdfText.rfind('positive sense')>0))

def singleStandedPositiveSense(pdfText):

    return (singleStranded(pdfText) and positiveSense(pdfText))

def sarsmerscoronacovid(pdfText):

    #return ((pdfText.rfind('sars')>0) or (pdfText.rfind('mers')>0) or (pdfText.rfind('corona')>0) or (pdfText.rfind('covid')>0))

    #may not be worth filtering for this criteria

    return True

def sarsmerscoronasingleStandedPositiveSense(pdfText):

    return (singleStranded(pdfText) and positiveSense(pdfText) and sarsmerscoronacovid(pdfText))

def rnaNucleoCapsid(txt):

    return ((txt.rfind('rna ')>0) and (txt.rfind('nucleoprotein')>0) and (txt.rfind('capsid')>0))

def matrixProtein(pdfText):

    return ((pdfText.rfind('matrix protein')>0) or (pdfText.rfind('matrix-protein')>0))

def spikeglycoProtein(pdfText):

    return ((pdfText.rfind('spike protein')>0) or (pdfText.rfind('glycoprotein')>0))

def humanActivator(pdfText):

    return ((pdfText.rfind('tmprss2')>0) or (pdfText.rfind('chromosome 21')>0) or (pdfText.rfind('21q22.3')>0))

def sarsSinglePositivestrandRNAnucleocapsidMatrixSpike(txt):

    return(sarsmerscoronasingleStandedPositiveSense(txt) and rnaNucleoCapsid(txt) and matrixProtein(txt) and spikeglycoProtein(txt))

def isDrugName(txt):

    toremovewords = ["vaccine","urine", "quarantine", "april","bovine","intestine", "examine", "decline", "determine","routine","canine", "hyaline", "polypeptide", "nucleotide", "epinephrine", "cytokine", "chemokine","line","swine","baseline","medicine","equine","serine","tyrosine","nine","murine","porcine","alanine","peptide","saline","define","feline","online","prost","creatinine","guideline","cysteine","norepinephrine","glutamine", "paracrine","adenosine","ovine","alkaline","telemedicine","endocrine","glycine","uterine","autocrine","pipeline","combine","machine","caprine","medline","midline","ukraine","caffeine","dourine", "fine","lysine","arginine","histamine"]

    #some of these words were when I was searching with drug suffix ine instead of quine

    if (txt in toremovewords):

        return False

    return True

def antibody47D11(pdfText):

    return (pdfText.rfind('antibody 47d11')>0 )





for stemAfix in drugStemsAffixes:

    maxPDFsForTesting = 150000

    pdfsProcessed = 0

    for shaid in shaToPDFText:

        txt = shaToPDFText[shaid]

        pdfsProcessed = pdfsProcessed + 1

        #print(pdfText)

        if (pdfsProcessed < maxPDFsForTesting):

            iterator=re.finditer(stemAfix,txt)

            for m in iterator:

                drugFound = shaToPDFText[shaid][shaToPDFText[shaid].rfind(' ',0, m.end()-2):m.end()]

                if ((len(drugFound) >= MIN_DRUG_LEN) and isDrugName(drugFound.rstrip().lstrip())):

                    if ( (antibody47D11(txt)) or ( sarsSinglePositivestrandRNAnucleocapsidMatrixSpike (txt) and humanActivator(txt) ) ):

                        drugsInPDFs.append(drugFound)

                if ((len(drugFound) >= MIN_DRUG_LEN) and isDrugName(drugFound.rstrip().lstrip())):

                    if ( (txt.rfind('cytokine storm')>0) or (txt.rfind('cytokine-storm')>0) ):

                        cytoDrugsInPDFs.append(drugFound)

                

drugs_set = list(set(drugsInPDFs))

cyto_drugs_set = list(set(cytoDrugsInPDFs))



count=[]

for d in drugs_set:

    count.append(-drugsInPDFs.count(d))

drugs_set=list(np.array(drugs_set)[np.array(count).argsort()]) 

print(len(drugs_set))



count=[]

for d in cyto_drugs_set:

    count.append(-cytoDrugsInPDFs.count(d))

cyto_drugs_set=list(np.array(cyto_drugs_set)[np.array(count).argsort()]) 

print(len(cyto_drugs_set))









import plotly.express as px

#Now lets count what drug has been mentioned the most

drugsDF = pd.DataFrame(drugs_set,columns=["Drug"])

cytodrugsDF = pd.DataFrame(cyto_drugs_set,columns=["Drug"])



def count1(drug,druglist):

    return druglist.count(drug)

drugsDF['CountInText'] = drugsDF.apply(lambda x: count1(x["Drug"],drugsInPDFs),axis=1) 

cytodrugsDF['CountInText'] = cytodrugsDF.apply(lambda x: count1(x["Drug"],cytoDrugsInPDFs),axis=1) 



#lets plot 10 most mentioned drugs

#MAXPLOT=10 

#plt.figure(figsize=(20,5))

#plt.bar(drugsDF["Drug"][(-drugsDF["CountInText"].to_numpy()).argsort()[:MAXPLOT]], drugsDF["CountInText"][(-drugsDF["CountInText"].to_numpy()).argsort()[:MAXPLOT]])

#plt.xticks(rotation=90,fontsize=12)

#plt.yticks(fontsize=12)

#plt.ylabel("Counts",fontsize=15)

#plt.title("Drug mentions")

#plt.show()

drugsDF["all"] = "all" #single root hack for plotly treemap

cytodrugsDF["cytoStorm"] = "cytoStorm" #single root hack for plotly treemap



drugsplot = px.treemap(drugsDF,path=['all', 'Drug'], values='CountInText')

drugsplot.show()



cytodrugsplot = px.treemap(cytodrugsDF,path=['cytoStorm', 'Drug'], values='CountInText')

cytodrugsplot.show()