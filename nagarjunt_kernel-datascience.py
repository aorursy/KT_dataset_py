

# Import required libraries

import os.path

from os.path import join, splitext

from tempfile import mkdtemp



import nltk, string

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

from werkzeug.utils import secure_filename





class ClassificationResultModel(object):

    """__init__() functions as the class constructor"""



    def __init__(self, idseq: object = None, business_description: object = None, original_accp_rej_status: object = None,

                 cosine_similarity_val: object = None) -> object:

        self.idseq = idseq

        self.business_description = business_description

        self.original_accp_rej_status = original_accp_rej_status

        self.cosine_similarity_val = cosine_similarity_val       



class ExecutionResultModel(object):

    def __init__(self, searchCompany: object = None, extractedBD: object = None, classifierResult: list = None) -> object:

        self.searchCompany = searchCompany

        self.extractedBD = extractedBD

        self.classifierResult = classifierResult



#------------------------------------------------------------



#=========================================================================================



def getDocumentsList(filepath, fileType):

    dataFile = None

    if(fileType == 'csv'):

        dataFile = pd.read_csv(filepath)

    else:

        dataFile = pd.read_excel(filepath)

    # Check for null values

    dataFile[dataFile.isnull().any(axis=1)]



    # Drop rows with null Values

    dataFile.drop(dataFile[dataFile.isnull().any(axis=1)].index,inplace=True)

    first_column = dataFile.columns[0]

    dataFile.drop([first_column], axis=1, inplace=True) # Dropping the uncessary column     

    return dataFile



stemmer = nltk.stem.porter.PorterStemmer()

remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)



def stem_tokens(tokens):

    return [stemmer.stem(item) for item in tokens]



'''remove punctuation, lowercase, stem'''

def normalize(text):

    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))



def executeForSim(inputCompName, inputSearchByBD, inputFilePath):

    # Import Data

    # inputFilePath = 'D:\\Nagarjun_DS\\dev\\tcapp\\dataset\\data_SD_CSV.csv'

    

    fileExt = inputFilePath.split(".")

    finalResult = ExecutionResultModel()

    if (len(fileExt) > 0 and (fileExt[1] == 'csv' or fileExt[1] == 'xls' or fileExt[1] == 'xlsx')):

        df = getDocumentsList(inputFilePath, fileExt[1])

        extractedBD = inputSearchByBD

        

        # df.head(10)

        # df.isnull().sum()

        # df.is_duplicate.value_counts()

        # 802/len(df)   - it's still good to establish some sort of a baseline. In this case, 93.25% will be our baseline for accuracy.



        # Use TfidfVectorizer() to transforms into vectors,

        # then compute their cosine similarity.

        vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

        Tfidf_scores = []

        for i in range(len(df)):

            tfidf = vectorizer.fit_transform([extractedBD, df.Comp_Business_Description[i]])

            roundOffCosmValue = round((((tfidf * tfidf.T).A)[0, 1]) * 100, 2)

            Tfidf_scores.append(ClassificationResultModel('', df.Comp_Business_Description[i], df.BD_Accept_Status[i],

                                                          str(roundOffCosmValue)))



        Tfidf_scores.sort(key=lambda x: float(x.cosine_similarity_val), reverse=True)

        i = 0

        for tempBD in Tfidf_scores:

            i = i + 1

            tempBD.idseq = str(i)

        finalResult = ExecutionResultModel(searchCompany=inputCompName, extractedBD=extractedBD, classifierResult=Tfidf_scores)

        return finalResult

    else:

        return finalResult







        
#inputBD = "Organization is engaged in providing computer peripherals, parts, memory cards, clinical instrument, medical lab items and clinical softwares. Main Income is from products and services."

#result = executeForSim('ABCD IT COMPANY', inputBD, 'D:\\Nagarjun_DS\\WorkingNotes\\Blogs\\data_SD_XLS.xlsx')

#print(result.searchCompany)

#print(result.extractedBD)

#print(result.classifierResult[0].business_description)

#print(result.classifierResult[0].original_accp_rej_status)

#print(result.classifierResult[0].cosine_similarity_val)