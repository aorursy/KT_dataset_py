import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



import nltk

import os

import re

import math



import json

import math

import glob



import spacy

from spacy.matcher import Matcher

from spacy.matcher import PhraseMatcher



from collections import defaultdict

from tqdm import tqdm



import ipywidgets as widgets



from IPython.display import Image

from IPython.display import display, HTML



nlp = spacy.load("en_core_web_sm")
# Reference:- https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30557-2/fulltext

virus_ref = ['covid-19', 'coronavirus', 'cov-2', 'sars-cov-2', 'sars-cov', 'hcov', ' hcov-19','2019-ncov']



def contains_virus(text):

    divider = "-"

    text_with_virus = re.findall(rf'({"|".join(virus_ref)})', text, flags=re.IGNORECASE)



    # Text should contain atleast one reference to the virus.

    if(len(text_with_virus) > 0):

        return True

    else:

        return False
def prepare_dataset():

    articles = {}



    for dirpath, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            if filename.endswith(".json"):

                articles[filename] = os.path.join(dirpath, filename)  



    root_dir = '/kaggle/input/CORD-19-research-challenge'



    df = pd.read_csv(f'{root_dir}/metadata.csv')



    # Drop duclicate duplicates.

    df = df.drop_duplicates(subset='abstract', keep="first")

    

    # Sort in descending order of publish time (most recent).

    df = df.sort_values(by = 'publish_time', ascending=False)

    

    # Remove na and replace with an empty string.

    df.sha.fillna("", inplace=True)



    literature = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        sha = str(row['sha'])

        if sha != "":

            sha = sha + '.json';

            try:

                found = False

                with open(articles[sha]) as f:

                    data = json.load(f)

                    for key in ['abstract', 'body_text']:

                        if found == False and key in data:

                            for content in data[key]:

                                text = content['text']

                                title = data['metadata']['title']

                                

                                # Remove citation, since it could cause problems in future analysis.

                                for citation in content['cite_spans']:

                                    text = text.replace(citation['text'], "")

                                

                                # Only include text that mentions any of the references of COVID-19

                                if contains_virus(text) == True: 

                                    id = data['paper_id'] 

                                    literature.append({'file': articles[sha], 'paperid': id, 'title' : title,'body': text})                                

            except KeyError:

                pass

    

    # Output is an array of all the articles that reference COVID-19 with their corresponding title and body.

    return literature
keywords = ["incubation period",

        "basic reproductive number", "basic reproduction number", "basic reproductive ratio", "basic reproduction ratio",

        "basic reproductive rate", "basic reproduction rate", "r0", "reproduction rate",

        "serial interval", "serial intervals", " chain of transmission",

        "modes of transmission", "transmission methods", "transmission modes", "transmission routes",

        "environmental factors", "environmental conditions"]



modes_of_transmission = ['direct contact', 'close contact', 'indirect contact', 'person contact',

    'respiratory droplets/route', 'airborne route', 'aerosols', 'airborne transmission', 

    'respiratory route','respiratory droplets', 'respiratory secretions',

    'surfaces/fomites', 'fomites', 'environmental surfaces', 'environment', 'virus spreads',

    'human transmission','fecal-oral', 'fecaloral route', 'faecaloral route']
# The first argument is the tokenizer.

# attr = 'LOWER' to make consistent capitalization.

keywords_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

transmission_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')



# A list of tokens is created for each of the terms list.

keywords_tokens_list = [nlp(item) for item in keywords]

transmission_tokens_list = [nlp(item) for item in modes_of_transmission]



# The rules are added to the matchers.

keywords_matcher.add("Keywords", None, *keywords_tokens_list)

transmission_matcher.add("Transmission_Modes", None, *transmission_tokens_list)
def keywords_match(text):

    return len(re.findall(rf'({"|".join(keywords)})', text)) > 0



def transmission_match(text):

    return len(re.findall(rf'({"|".join(modes_of_transmission)})', text)) > 0
def find_matches_per_document():

    # Integers that represent the total number of matches per document.

    size_keywords_match = 0

    size_transmission_match = 0

    

    # A dictionary that contains every term and its corresponding frequency within the current document.

    keyword_frequencies = {}

    transmission_frequencies = {}

    

    #  A dictionary that contains every term and all its found excerpts within the current document.

    keyword_excerpts = defaultdict(list)

    transmission_excerpts = defaultdict(list)

    

    # An array the contains all the found sentences for a certain term.

    found_sentences = []

    found_sentences2 = []

    

    # A dictionary that contains all the information needed for each term found.

    terms_info = defaultdict(list)

    terms_info2 = defaultdict(list)

    

    # A dictionary with every documents details, total number of matches and the corresponding term matches information.

    keywords_per_document = defaultdict(list)

    transmission_per_document = defaultdict(list)

 

    allow = False

    

    # Prepare dataset and retrieve the relevant articles.

    literature = prepare_dataset()



    for article in tqdm(literature):

        paperid = article['paperid']

        title = article['title']

        text_list = re.compile("\. ").split(article['body'])



        for text in text_list:     

            if callable(keywords_match):

                    allow = keywords_match(text)

            if allow == False:

                    allow = transmission_match(text)

            if allow == True: 

                text = text.lower()

                doc = nlp(text)



                matches_keywords = keywords_matcher(doc)

                matches_trasmission = transmission_matcher(doc)



                size_keywords_match = size_keywords_match + len(matches_keywords) 

                size_transmission_match = size_transmission_match + len(matches_trasmission)



                # Create a set of the items found in the text.

                found_keywords = [{"term":doc[match[1]:match[2]], "match_excerpt":doc[match[1]-10:match[2]+100]}

                           for match in matches_keywords]



                found_transmission = [{"term":doc[match[1]:match[2]], "match_excerpt":doc[match[1]-10:match[2]+100]}

                           for match in matches_trasmission]

                

                # Computer term frequencies for the found terminologies in the keywords list.

                for item in found_keywords:

                    if str(item["term"]).lower() in keyword_frequencies:

                          keyword_frequencies[str(item["term"]).lower()] += 1

                    else:

                          keyword_frequencies[str(item["term"]).lower()] = 1

                

                # Compute term frequencies for the found terminologies in the modes_of_transmission.

                for item in found_transmission:

                    if str(item["term"]).lower() in transmission_frequencies:

                          transmission_frequencies[str(item["term"]).lower()] += 1

                    else:

                          transmission_frequencies[str(item["term"]).lower()] = 1

                

                # Collect all excerpts for each term.

                for item in found_keywords:

                    keyword_excerpts[str(item["term"]).lower()].append({"match_excerpt":item["match_excerpt"]})



                for item in found_transmission:

                    transmission_excerpts[str(item["term"]).lower()].append({"match_excerpt":item["match_excerpt"]})

                    

        for term in keyword_frequencies:

            for item in keyword_excerpts.items():

                if(term == item[0]):

                    for i in item[1]:

                        found_sentences.append(i["match_excerpt"])



            terms_info[str(term).lower()].append({"matches": keyword_frequencies[term], "sentences" : found_sentences})

            found_sentences = []



        for term in transmission_frequencies:

            for item in transmission_excerpts.items():

                if( term == item[0]):

                    for i in item[1]:

                        found_sentences2.append(i["match_excerpt"])



            terms_info2[str(term).lower()].append({"matches": transmission_frequencies[term], "sentences" : found_sentences2})

            found_sentences2 = []

        

        # If the document contains any matches, we store all their details.

        if( size_keywords_match != 0):

            keywords_per_document[paperid].append({"title" : title, "matches": size_keywords_match, "terms_info": terms_info, "body_text": text})

            size_keywords_match = 0

            keyword_frequencies = {}



        if( size_transmission_match != 0):

            transmission_per_document[paperid].append({"title" : title, "matches": size_transmission_match, "terms_info": terms_info2, "body_text": text})

            size_transmission_match = 0

            transmission_frequencies = {}



        keyword_excerpts = defaultdict(list)

        transmission_excerpts = defaultdict(list)

        

        terms_info = defaultdict(list)

        terms_info2 = defaultdict(list)

        

    return keywords_per_document, transmission_per_document
def plotIncubationDays():

    global first5

    global second5

    global third5

    global forth5

    global fifth5

    global sixth5

    global seventh5

    

    x_axis = ('1-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+')

    data = [first5,second5,third5,forth5,fifth5,sixth5,seventh5]



    plt.bar(x_axis, data, align='center', alpha=0.5)

    #plt.xticks(x_axis, data)

    plt.ylabel('Number of articles')

    plt.xlabel('Number of days')



    display(HTML("""

    <style>

    .output {

        display: flex;

        align-items: center;

        text-align: center;

    }

    </style>

    """))



    display(HTML(f'<h2>{"Incubation Period"}</h2>'))



    plt.show()

    

def plotTranmissionModes():

    global contact

    global respiratory

    global surfaces

    global fecaoral

    global human 

    

    height = [contact, respiratory, surfaces, fecaoral, human]

    bars = ('Contact', 'Respiratory/droplets', 'Surfaces/fomites', 'Fecaoral routes', 'Human transmission')

    y_pos = np.arange(len(bars))

    

    plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.6))

    plt.xticks(y_pos, bars)

    plt.ylabel('Number of articles')

    plt.xticks(rotation=55)

    

    display(HTML("""

    <style>

    .output {

        display: flex;

        align-items: center;

        text-align: center;

    }

    </style>

    """))



    display(HTML(f'<h2>{"Transmission Modes"}</h2>'))

    

    plt.show()
# Global terms used to cound the values needed for plotting.

first5 = 0

second5 = 0

third5 = 0

forth5 = 0

fifth5 = 0

sixth5 = 0

seventh5 = 0



contact = 0

respiratory = 0

surfaces = 0

fecaoral = 0

human = 0
def withinRange(number, min, max): 

    if int(number) < min or int(number) > max:

        return False

    

    return True



def isfloat(string):

    try:

        float(string)

        return True

    except ValueError:

        return False



def extractNumbers(incubationPeriod):

    array_of_numbers = [int(i) for i in incubationPeriod.split() if i.isdigit()]

        

    if array_of_numbers == []:

        array_of_numbers = [round(float(i)) for i in incubationPeriod.split() if isfloat(i)]



    if array_of_numbers == []:

        if '-' in incubationPeriod:

            array_of_numbers = [int(i.split('-')[1]) for i in incubationPeriod.split() if i.split('-')[0].isdigit() and i.split('-')[1].isdigit() ]



    return array_of_numbers



def common_transmission_modes(transmissionSentence):

    global contact

    global respiratory

    global surfaces

    global fecaoral

    global human 

    

    if "contact" in transmissionSentence:

        contact += 1

    if "respiratory" in transmissionSentence or "airborne" in transmissionSentence or "aerosols" in transmissionSentence:

        respiratory += 1

    if "surfaces" in transmissionSentence or "fomites" in transmissionSentence:

        surfaces += 1

    if "fecal-oral" in transmissionSentence or "fecaloral" in transmissionSentence or "faecaloral" in transmissionSentence:

        fecaoral += 1

    if "human transmission" in transmissionSentence:

        human += 1



def chartData(array_of_numbers):

    global first5

    global second5

    global third5

    global forth5

    global fifth5

    global sixth5

    global seventh5

    

    for num in array_of_numbers:

        if withinRange(num, 1,5):

            first5 += 1

        elif withinRange(num, 5,10):

            second5 += 1

        elif withinRange(num, 10,15):

            third5 += 1

        elif withinRange(num, 15,20):

            forth5 += 1

        elif withinRange(num, 20,25):

            fifth5 += 1

        elif withinRange(num, 25,30):

            sixth5 += 1

        elif withinRange(num, 30,100):

            seventh5 += 1
def extract_relevant_sentences():

    incubationOutput = []

    reproductionOutput = []

    serialOutput = []

    environmentalOutput = []



    incubationPeriod = ""

    reproductionSentence = ""

    environmentalSentence = ""

    serialSentence = ""

    

    global third5

    

    keywords_per_document, tranmission_per_document = find_matches_per_document()

    

    for paperid in keywords_per_document.items():

        for data in paperid[1]:

            for term in data['terms_info'].items():



                if "incubation" in str(term[0]):

                    for array in term[1]:

                        for item in array['sentences']:

                            if "days" in str(item) and "incubation" in str(item):

                                incubationPeriod = str(item)

                                break;

                            elif "weeks" in str(item) and "incubation" in str(item):

                                incubationPeriod = str(item)

                                third5 += 1

                                break;



                elif "r0" in str(term[0]) or "r 0" in str(term[0]) or "reproductive" in str(term[0]) or "reproduction" in str(term[0]):

                       for array in term[1]:

                            for item in array['sentences']:

                                for char in item:

                                    if (str(char) != "r0" and str(char) != "0") and (str(char).isdigit() or isfloat(str(char))):

                                        if "r0" in str(item) or "r 0" in str(item) or "reproductive" in str(item) or "reproduction" in str(item):

                                            reproductionSentence = str(item)

                                            break



                elif "serial" in str(term[0]):

                       for array in term[1]:

                            for item in array['sentences']:

                                if "days" in str(item) and "serial" in str(item):

                                    serialSentence = str(item)

                                    break



                elif "environmental" in str(term[0]):

                       for array in term[1]:

                            for item in array['sentences']:

                                if "environmental" in str(item) and ("such as" in str(item) or "like" in str(item)):

                                    environmentalSentence = str(item)

                                    break





        if incubationPeriod != "":

            res = extractNumbers(incubationPeriod)



            if res != []:

                incubationOutput.append([data['title'], data['matches'], incubationPeriod])

                incubationPeriod = ""

                chartData(res)



        if reproductionSentence != "":

            reproductionOutput.append([data['title'], data['matches'], reproductionSentence])

            reproductionSentence = ""



        if serialSentence != "":

            serialOutput.append([data['title'], data['matches'], serialSentence])

            serialSentence = ""



        if environmentalSentence != "":

            environmentalOutput.append([data['title'], data['matches'], environmentalSentence])

            environmentalSentence = ""

            

    return incubationOutput, reproductionOutput, serialOutput, environmentalOutput 
def transmission_routes():

    transmissionOutput = []

    transmissionSentence = ""

    

    keywords_per_document, tranmission_per_document = find_matches_per_document()



    for paperid in tranmission_per_document.items():

        for data in paperid[1]:

            if data['title'] != "":

                for term in data['terms_info'].items():

                    for array in term[1]:

                        for item in array['sentences']:

                            transmissionSentence = str(item)

                            break



        if transmissionSentence != "": 

            transmissionOutput.append([data['title'], data['matches'], transmissionSentence])

            common_transmission_modes(transmissionSentence)

            transmissionSentence = ""

        

    return transmissionOutput
def output_results(array, term, term2, term3, highlight):

    count = 0

    for item in array:

        if count == 4:

            break

            

        if(item[0] != "" and item[0] != "comment" and item[0] != "authors"):

            sentence = ""

            count += 1

            if highlight == True:

                if term in item[2]:

                    sentence = str(item[2]).replace(term, f"<mark><b>{term}</b></mark>")

                elif term2 != "" and term2 in item[2]:

                    sentence = str(item[2]).replace(term2, f"<mark><b>{term2}</b></mark>")

                elif term3 != "" and term3 in item[2]:

                    sentence = str(item[2]).replace(term3, f"<mark><b>{term3}</b></mark>")

            else:

                sentence = str(item[2])

                

            display(HTML(f"""



                    <p>

                        <strong>Article Title:</strong> {item[0]}

                    </p>

                    <p>

                        <strong>Total number of matches:</strong> {item[1]}

                    </p>

                    <blockquote>...{sentence}...</blockquote>"""))
# function to return the second element of the two elements passed as the parameter 

def sortSecond(val): 

    return val[1]  



incubationOutput, reproductionOutput, serialOutput, environmentalOutput = extract_relevant_sentences()

transmissionOutput = transmission_routes()



# Arrays sorted according to the number of matches in descending order. 

incubationOutput.sort(key = sortSecond, reverse = True) 

reproductionOutput.sort(key = sortSecond, reverse = True)

serialOutput.sort(key = sortSecond, reverse = True)

environmentalOutput.sort(key = sortSecond, reverse = True)

transmissionOutput.sort(key = sortSecond, reverse = True)
plotIncubationDays()
plotTranmissionModes()
output_results(incubationOutput, "incubation period", "", "", True)
output_results(reproductionOutput, "reproduction number", "reproductive number", "r0", True)
output_results(serialOutput, "serial interval", "", "", True)
output_results(environmentalOutput, "environmental factors", "environmental conditions", "", True)
output_results(transmissionOutput, "", "", "", False)