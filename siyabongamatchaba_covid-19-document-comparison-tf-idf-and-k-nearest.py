import tarfile
import numpy as np
import pandas as pd
import json
 
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
import re, string, timeit
import os
from math import sqrt
import pickle
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
covidTerms = ['sars-cov-2','covid-19', 'coronavirus']
incubationTerms = ['incubation period', 'incubation','incubations','incubation time']
punctuationTerms = ['!','(',')','-','[',']','{','}',';',':','\ ', '<', '>','/','?','@','#','$','%','^','&','*','_','~','+','=']

monthTerms = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']



countries = ["Afghanistan","Albania","Algeria","Andorra","Angola","Antigua & Deps","Argentina","Armenia","Australia","Austria","Azerbaijan"
,"Bahamas"
"Bahrain",
"Bangladesh",
"Barbados",
"Belarus",
"Belgium",
"Belize",
"Benin",
"Bhutan",
"Bolivia",
"Bosnia Herzegovina",
"Botswana",
"Brazil",
"Brunei",
"Bulgaria",
"Burkina",
"Burundi",
"Cambodia",
"Cameroon",
"Canada",
"Cape Verde",
"Central African Rep",
"Chad",
"Chile",
"China",
"Colombia",
"Comoros",
"Congo",
"Congo {Democratic Rep}",
"DRC",
"Costa Rica",
"Croatia",
"Cuba",
"Cyprus",
"Czech Republic",
"Denmark",
"Djibouti",
"Dominica",
"Dominican Republic",
"East Timor",
"Ecuador",
"Egypt",
"El Salvador",
"Equatorial Guinea",
"Eritrea",
"Estonia",
"Ethiopia",
"Fiji",
"Finland",
"France",
"Gabon",
"Gambia",
"Georgia",
"Germany",
"Ghana",
"Greece",
"Grenada",
"Guatemala",
"Guinea",
"Guinea-Bissau",
"Guyana",
"Haiti",
"Honduras",
"Hungary",
"Iceland",
"India",
"Indonesia",
"Iran",
"Iraq",
"Ireland",
"Israel",
"Italy",
"Ivory Coast",
"Jamaica",
"Japan",
"Jordan",
"Kazakhstan",
"Kenya",
"Kiribati",
"Korea North",
"Korea South",
"Korea",
"Kosovo",
"Kuwait",
"Kyrgyzstan",
"Laos",
"Latvia",
"Lebanon",
"Lesotho",
"Liberia",
"Libya",
"Liechtenstein",
"Lithuania",
"Luxembourg",
"Macedonia",
"Madagascar",
"Malawi",
"Malaysia",
"Maldives",
"Mali",
"Malta",
"Marshall Islands",
"Mauritania",
"Mauritius",
"Mexico",
"Micronesia",
"Moldova",
"Monaco",
"Mongolia",
"Montenegro",
"Morocco",
"Mozambique",
"Myanmar",
"Burma",
"Namibia",
"Nauru",
"Nepal",
"Netherlands",
"New Zealand",
"Nicaragua",
"Niger",
"Nigeria",
"Norway",
"Oman","Pakistan","Palau","Panama","Papua New Guinea","Paraguay","Peru","Philippines","Poland","Portugal","Qatar","Romania","Russian Federation","Russia","Rwanda","St Kitts & Nevis","St Lucia","Saint Vincent & the Grenadines","Samoa","San Marino","Sao Tome & Principe","Saudi Arabia","Senegal","Serbia","Seychelles","Sierra Leone","Singapore","Slovakia","Slovenia","Solomon Islands","Somalia","South Africa","South Sudan","Spain","Sri Lanka","Sudan","Suriname","Swaziland","Sweden","Switzerland","Syria","Taiwan","Tajikistan","Tanzania","Thailand","Togo","Tonga","Trinidad & Tobago","Tunisia","Turkey","Turkmenistan","Tuvalu","Uganda","Ukraine","United Arab Emirates","UAE","United Kingdom","UK","United States","US","Uruguay","Uzbekistan","Vanuatu","Vatican City","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe"]
#load files
def loadZip(file):
    json_files = [pos_json for pos_json in os.listdir(file) if pos_json.endswith('.json')]

    
    
    return json_files
contentList = loadZip('/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset')
contentListNonCom = loadZip('/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset')
#biorxiv_medrxiv dataset + PMCCusomt
contentListBio = loadZip('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv')
contentListPmc = loadZip('/kaggle/input/CORD-19-research-challenge/custom_license/custom_license')
#combine articles into single string for tokenization
def text2pandas(ls,base):
    

    articles = []
    uniqueArticles = []
    for x in ls:
        dataList = ["","",'']
        with open(base+x) as f:
          
            d = json.loads(f.read())
            dataList[0] = d['paper_id']
            dataList[1] = d['metadata']['title']
            string = ""
            for n in d['body_text']:
                string += n['text']
            dataList[2] = string
            uniqueArticles.append(string)
            articles.append(dataList)
        
    dataFrame = pd.DataFrame(articles, columns = ['ID','Title', 'Corpus']) 
        
    
    return uniqueArticles,dataFrame
#turn data into panda for bio and pmc data
articlesBio,dfBio = text2pandas(contentListBio,"/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/")
articlesPmc, dfPmc = text2pandas(contentListPmc,"/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/")
#turn data into panda for com data
articles,df = text2pandas(contentList,"/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/")

#turn data into panda for noncom data
articlesNonCom,dfNonCom = text2pandas(contentListNonCom,"/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/")
del dfNonCom
del df
del dfBio
del dfPmc

#create tokens from article corpus
def createTokenList(articles):
    articleTokens = []
    for x in articles:
        tokens1 = nltk.word_tokenize(x)
        
        articleTokens.append(tokens1)
    return articleTokens

#coprus com zip tokens
articleTokens = createTokenList(articles)
#coprus non-com zip tokens
articleTokensNonCom = createTokenList(articlesNonCom)
#corpus bio and pmc tokens
articleTokensBio = createTokenList(articlesBio)
articleTokensPmc = createTokenList(articlesPmc)
#flatten token list

flatten = lambda l: [item for sublist in l for item in sublist]
#flattenTokenList = (flatten(articleTokens))
flattenTokenList = (flatten(articleTokens))
#flatten Non Com token list
flattenTokenListNonCom = (flatten(articleTokensNonCom))
#flatten bio and pmc
flattenTokenListBio = (flatten(articleTokensBio))
flattenTokenListPmc = (flatten(articleTokensPmc))
del articleTokens
del articleTokensNonCom
del articleTokensBio
del articleTokensPmc
#word tag list
def dateCheck(string):
    listofDates = 0
    for i in range(len(string)):
        if(bool(re.match('^[0-3][0-9][0-9][0-9]', string[i]))) == True:
            listofDates= listofDates + 1
            
            string[i] = "<<year>>"
        
    return string;
def stopwordCheck(string):
    """Remove stop words from list of tokenized words"""
    newStopWords =0
    for i in range(len(string)):
        if string[i] in stopwords.words('english'):
            newStopWords = newStopWords + 1
            string[i] = "<<stopWord>>"
    return string
def numberCheck(string):
    """Remove numbers from list of tokenized words"""
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    new_words = 0
    for i in range(len(string)):
        finalSt = ""
        for charac in string[i]:
            if charac not in punctuations:
                finalSt = finalSt + charac

        if bool(re.match('^[0-9]', finalSt)) == True:
            new_words = new_words + 1
            string[i] = "<<realNumber>>"
    return string;
def countryCheck(string):

    lss = []
    for i in range(len(string)):
        if string[i] in countries:
           lss.append(string[i])
           string[i] = "<<country>>"
            

    return lss
def covidCheck(string):

    lss = []
    for i in range(len(string)):
        if string[i].lower() in covidTerms:
           lss.append(string[i])
           string[i] = "<<COVID-19>>"
            

    return string
def punctuationCheck(string):

    lss = []
    for i in range(len(string)):
        if string[i].lower() in punctuationTerms:
           lss.append(string[i])
           string[i] = "<<Punctuation>>"
            

    return string
def monthCheck(string):

    lss = []
    for i in range(len(string)):
        if string[i].lower() in monthTerms:
           lss.append(string[i])
           string[i] = "<<Month>>"
            

    return string


#replace list of words to list of tokens
def integer_representations(vocab_dict,token_list):
    integer_rep = []
    for token in token_list:
        try:
          
          integer_rep.append((vocab_dict[token])[0])
        except:
          integer_rep.append(999999999)
    return integer_rep
#gets rid of words that do not appear more than the set frequency threshold
def useless_words(freq_dict, freq_threshold):
    copy = freq_dict.copy()
    for word in copy:
        if freq_dict[word][1] < freq_threshold:
            del freq_dict[word]
    return freq_dict

#Com Corpus
newStopsList = stopwordCheck(flattenTokenList)
newList = punctuationCheck(newStopsList)
newList = monthCheck(newList)
newList = covidCheck(newList)
newList = dateCheck(newList)
newList = numberCheck(newList)
#non com corpus
newStopsListNon = stopwordCheck(flattenTokenListNonCom)
newListNon = punctuationCheck(newStopsListNon)
newListNon = monthCheck(newListNon)
newListNon = covidCheck(newListNon)
newListNon = dateCheck(newListNon)
newListNon = numberCheck(newListNon)
#bio corpus
newStopsListBio = stopwordCheck(flattenTokenListBio)
newListBio = punctuationCheck(newStopsListBio)
newListBio = monthCheck(newListBio)
newListBio = covidCheck(newListBio)
newListBio = dateCheck(newListBio)
newListBio = numberCheck(newListBio)
#pmc corpus
newStopsListPmc = stopwordCheck(flattenTokenListPmc)
newListPmc = punctuationCheck(newStopsListPmc)
newListPmc = monthCheck(newListPmc)
newListPmc = covidCheck(newListPmc)
newListPmc = dateCheck(newListPmc)
newListPmc = numberCheck(newListPmc)
def frequency2(newList):
    d = {}
    wID = 0
    for t in newList:
        try:
            elem = d[t]
        except:
            elem = [wID,0]
            wID = wID + 1
        elem[1] = elem[1] + 1
        d[t] = elem
    return d
def frequencyUpdate(newList,d):
    wID = 0
    for t in newList:
        try:
            elem = d[t]
        except:
            elem = [wID,0]
            wID = wID + 1
        elem[1] = elem[1] + 1
        d[t] = elem
    return d
freqDictionary = frequency2(newList)
newDictionary = frequencyUpdate(newListNon,freqDictionary)
newDictionary2 = frequencyUpdate(newListBio,newDictionary)
finalDictionary = frequencyUpdate(newListPmc,newDictionary2)
output = open('comboDictionary.pkl', 'wb')
pickle.dump(finalDictionary, output)
output.close()

finalDictionary = pd.read_pickle("/kaggle/input/covidsaved/comboDictionary.pkl")
freq_threshold = 3
newDictValues = useless_words(finalDictionary, freq_threshold)
def finalDictionary(dictionary):
    copy = dictionary.copy()
    for word in copy:
        del dictionary[word][1]
    return dictionary
final_dict = finalDictionary(newDictValues)
#turn articles in tokens
def newPDList(articles,dictionary):
    
    tokenFinalList = []
    for x in articles:
        tokens1 = nltk.word_tokenize(x)
        copyCheck = tokens1.copy()
        punct = punctuationCheck(copyCheck)
        covid = covidCheck(copyCheck)
        dateC = dateCheck(copyCheck)
        ls = stopwordCheck(copyCheck)
        newW = numberCheck(copyCheck)
        countryL = countryCheck(copyCheck)
        month = monthCheck(copyCheck)
        
        #used dictionary to alter string tokens to integers
        newList = integer_representations(dictionary,copyCheck)
        tokenFinalList.append(newList)
    return tokenFinalList
#create com corpus token list from full dictionary
newArticleList = newPDList(articles,final_dict)
#create non com corpus token list from full dictionary
newArticleListNonCom = newPDList(articlesNonCom,final_dict)
#create pmc corpus token list from full dictionary
newArticleListPmc = newPDList(articlesPmc,final_dict)

#create bio corpus token list from full dictionary
newArticleListBio = newPDList(articlesBio,final_dict)
#save com article tokens
output = open('trainingTokensCom.pkl', 'wb')
pickle.dump(newArticleList, output)
output.close()
#save non com article tokens
output = open('trainingTokensNonCom.pkl', 'wb')
pickle.dump(newArticleListNonCom, output)
output.close()
#save pmc article tokens
output = open('trainingTokensPmc.pkl', 'wb')
pickle.dump(newArticleListPmc, output)
output.close()
#save bio article tokens
output = open('trainingTokensBio.pkl', 'wb')
pickle.dump(newArticleListBio, output)
output.close()
def addToPanda(text,newArticleList,base):
    

    articles = []
    count = 0
    for x in text:
        dataList = ["","",'']
        with open(base+x) as f:
          
            d = json.loads(f.read())
            dataList[0] = d['paper_id']
            dataList[1] = d['metadata']['title']
            string = ""
            for n in d['body_text']:
                string += n['text']
            dataList[2] = newArticleList[count]
            dataList[3] = string
            articles.append(dataList)
            count = count + 1
        
    dataFrame = pd.DataFrame(articles, columns = ['ID','Title', 'Corpus', 'CorpusText']) 
    length = []
    [length.append(len(str(text))) for text in dataFrame['Corpus']]
    dataFrame['Length'] = length
    return dataFrame
#data frame of first zip
newDf = addToPanda(contentList,newArticleList,"/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/")
#data frame of pmc zip
newDfPmc = addToPanda(contentListPmc, newArticleListPmc,"/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/")
#data frame if bio zip
newDfBio = addToPanda(contentListBio, newArticleListBio,"/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/")
newDfNonCom = addToPanda(contentListNonCom, newArticleListNonCom,"/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/")
#concatenate tables to form one table
frames = [newDf, newDfPmc, newDfBio, newDfNonCom]

result = pd.concat(frames)
#check panda frame to ensure corpus of integers
result.head()
#check smallest and largest article lenghts
min(result['Length']), max(result['Length']), round(sum(result['Length'])/len(result['Length']))
#remove articles with lengths less than 5000 tokens
result2 = result.copy()
result2 = result2.drop(result2['Corpus'][result2['Length'] < 5000].index, axis = 0)


#check again for any discrepencies in legnth
min(result2['Length']), max(result2['Length']), round(sum(result2['Length'])/len(result2['Length']))
#save final table
output = open('trainingFinal.pkl', 'wb')
pickle.dump(result2, output)
output.close()
result2 = pd.read_pickle("/kaggle/input/covidsaved/trainingFinal (1).pkl")

from sklearn.metrics.pairwise import cosine_similarity
def tfIdfSearch(docs,docTitles,title,num_neighbors,df4):
  try:
    test_row = (df4.loc[df4['Title'] == title]['CorpusText'].tolist())[0]
  except:
    test_row = title
    title = "your text"
  
  docs= docs + (test_row,)
  docTitles= docTitles + (title,)
  tfidf_vectorizer = TfidfVectorizer()
  tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
  l = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix)
  newDt = {}
  counting = 0

  for x in l[0]:
      newDt[x] = {docTitles[counting]:docs[counting]}
      counting = counting + 1
  neighborList = []

  neighCount = 0
  for i in sorted (newDt,reverse=True): 
          if neighCount != num_neighbors:
              neighborList.append(("distance: " + str(i),newDt[i]))
              neighCount = neighCount + 1
          elif neighCount >= neighCount:
              return neighborList

#Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    #pad empty columns
    
    if len(row1) < len(row2):
      N = len(row2) - len(row1)
      for x in range(N):
        row1.append(0)
    
    elif len(row1) > len(row2):
      N = len(row1) - len(row2)
      for x in range(N):
        row2.append(0)
    for i in range(len(row2)-1):
        
       
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)
def cosineSim(row1,row2):
  #pad empty columns
  
  if len(row1) < len(row2):
    N = len(row2) - len(row1)
    for x in range(N):
      row1.append(0)
    
  elif len(row1) > len(row2):
    N = len(row1) - len(row2)
    for x in range(N):
      row2.append(0)
  
  result = 1 - spatial.distance.cosine(row1, row2)
  
  return result
  
#Find the closest neighbors
def get_neighborsTitle(title,train, test_row, num_neighbors,trainTxt,df4,cosEu):
    distances = list()
    distDict = {}
    newCount = 0
    test_row = (df4.loc[df4['Title'] == test_row]['Corpus'].tolist())[0]
    if cosEu == "cosine":
      for train_row in (train):
        if train_row != test_row:
            dist = cosineSim(test_row, train_row)
            distances.append((train_row, "distance "+ str(dist)))
            distDict[dist] = {title.iloc[newCount]:trainTxt.iloc[newCount]}
        newCount = newCount + 1
    else:
      for train_row in (train):
        if train_row != test_row:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, "distance "+ str(dist)))
            distDict[dist] = {title.iloc[newCount]:trainTxt.iloc[newCount]}
        newCount = newCount + 1
    
        
    distances.sort(key=lambda tup: tup[1])
    neighborList = []
    neighCount = 0
    for i in sorted (distDict): 
        if neighCount != num_neighbors:
            neighborList.append((distDict[i],"distance: " + str(i)))
            neighCount = neighCount + 1
        elif neighCount >= neighCount:
            return neighborList
def transformText(text,dictionary):
  tokens1 = nltk.word_tokenize(text)
  copyCheck = tokens1.copy()
  punct = punctuationCheck(copyCheck)
  covid = covidCheck(copyCheck)
  dateC = dateCheck(copyCheck)
  ls = stopwordCheck(copyCheck)
  newW = numberCheck(copyCheck)
  countryL = countryCheck(copyCheck)
  month = monthCheck(copyCheck)
  #retrive unique vocab list
  #uniqueLs = get_vocabs(copyCheck)
  #vocabDict = dictionaryM(uniqueLs)
  #used dictionary to alter string tokens to integers
  newList = integer_representations(dictionary,copyCheck)
  return newList
def get_neighborsText(title,train, test_row, num_neighbors,dictionary,trainTxt,cosEu):
    distances = list()
    distDict = {}
    newCount = 0
    test_row = transformText(test_row,dictionary)
    if cosEu == "cosine":
      for train_row in (train):
        if train_row != test_row:
            dist = cosineSim(test_row, train_row)
            distances.append((train_row, "distance "+ str(dist)))
            distDict[dist] = {title.iloc[newCount]:trainTxt.iloc[newCount]}
        newCount = newCount + 1
    else:
      for train_row in (train):
        if train_row != test_row:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, "distance "+ str(dist)))
            distDict[dist] = {title.iloc[newCount]:trainTxt.iloc[newCount]}
        newCount = newCount + 1
        
    distances.sort(key=lambda tup: tup[1])
    neighborList = []
    neighCount = 0
    for i in sorted (distDict): 
        if neighCount != num_neighbors:
            neighborList.append(("distance: " + str(i),distDict[i]))
            neighCount = neighCount + 1
        elif neighCount >= neighCount:
            return neighborList
#build corpus text tuple and title text tuple for tf idf calculations
documents = (

)
documentTitles = ()
for x in range(len(result2)):
  documents = documents + (result2.iloc[x]['CorpusText'],)
  documentTitles = documentTitles + (result2.iloc[x]['Title'],)


#save save document tuples
output = open('trainingDocTuples.pkl', 'wb')
pickle.dump(documents, output)
output.close()
#save save document tuples
output = open('trainingDocTitleTuples.pkl', 'wb')
pickle.dump(documentTitles, output)
output.close()
documents = pd.read_pickle("/kaggle/input/covidsaved/trainingDocTuples (1).pkl")
documentTitles = pd.read_pickle("/kaggle/input/covidsaved/trainingDocTitleTuples (1).pkl")


txt1 = "Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery."
txt2 = "Prevalence of asymptomatic shedding and transmission (particularly children)."
txt3 = "Seasonality of transmission."
txt4 = "Physical science of the coronavirus ( charge distribution, adhesion to hydrophilic or phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding)."
txt5 = "Persistence and stability on a multitude of substrates and sources (nasal discharge, sputum, urine, fecal matter, blood)."
txt6 = "Persistence of virus on surfaces of different materials (copper, stainless steel, plastic)."
txt7 = "Natural history of the virus and shedding of it from an infected person"
txt8 = "Implementation of diagnostics and products to improve clinical processes"
txt9 = "Disease models, including animal models for infection, disease and transmission"
txt10 = "Tools and studies to monitor phenotypic change and potential adaptation of the virus"
txt11 = "Immune response and immunity"
txt12 = "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings"
txt13 = "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings"
txt14 = "Role of the environment in transmission"
riskFactors = ["Data on potential risks factors","Smoking, pre-existing pulmonary disease","Co-infections (determine whether co-existing respiratory or viral infections make the virus more transmissible or virulent) and other co-morbidities Neonates and pregnant women","Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.","Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors","Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups","Susceptibility of populations","Public health mitigation measures that could be effective for control"]
infoShare1 = "Methods for coordinating data-gathering with standardized nomenclature."
infoShare2 = "Sharing response information among planners, providers, and others."
infoShare3 = "Understanding and mitigating barriers to information-sharing."
infoShare4 = "How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic)."
infoShare5 = "Integration of federal/state/local public health surveillance systems."
infoShare6 = "Value of investments in baseline public health response infrastructure preparedness"
infoShare7 = "Modes of communicating with target high-risk populations (elderly, health care workers)."
infoShare8 = "Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too)."
infoShare9 = "Communication that indicates potential risk of disease to all population groups."
infoShare10 = "Misunderstanding around containment and mitigation."
infoShare11 = "Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment."
infoShare12 = "Measures to reach marginalized and disadvantaged populations."
infoShare13 = "Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities."
infoShare14 = "Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment."
infoShare15 = "Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care"
considerations1 = "Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019"
considerations2 = "Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight"
considerations3 = "Efforts to support sustained education, access, and capacity building in the area of ethics"
considerations4 = "Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences."
considerations5 = "Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)"
considerations6 = "Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed."
considerations7 = "Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media."
surveil1 = "How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs)."
surveil2 = "Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms."
surveil3 = "Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues."
surveil4 = "National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public)."
surveil5 = "Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy."
surveil6 = "Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded)."
surveil7 = "Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices."
surveil8 = "Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes."
surveil9 = "Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling."
surveil10 = "Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions."
surveil11 = "Policies and protocols for screening and testing."
surveil12 = "Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents."
surveil13 = "Technology roadmap for diagnostics."
surveil14 = "Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment."
surveil15 = "New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases."
surveil16 = "Coupling genomics and diagnostic testing on a large scale."
surveil17 = "Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant."
surveil18 = "Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional."
surveil19 = "One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors."
txt = "Your Text"
title = "Late viral or bacterial respiratory infections in lung transplanted patients: impact on respiratory function"
#here you can replace txt13 with your text in txt or by an existing title

neighborCount = 4
neighborList = tfIdfSearch(documents,documentTitles,txt13,neighborCount,result2)
neighborList
#enter in what you want to search for in text
text2 = "Your Text"
text = txt1
neighborCount = 4
neighbors = get_neighborsText(result2['Title'],result2['Corpus'], text, neighborCount,final_dict,result2['CorpusText'],"euclidean")
print("original: ")
print(text)

#Each new neighbor represents a similar article

for neighbor in neighbors:
    print('newNeighbor')
    print(neighbor)
#enter in article title 
title = "A Mini-Review on the Epidemiology of Canine Parvovirus in China"
neighborCount = 4

neighbors = get_neighborsTitle(result2['Title'],result2['Corpus'], title, neigborCount,result2['CorpusText'],result2,'cosine')
print("original title: ")
print(title)

for neighbor in neighbors:
    print('newNeighbor')
    print(neighbor)