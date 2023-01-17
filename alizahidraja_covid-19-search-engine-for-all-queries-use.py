import pandas as pd

import numpy as np

import os 

import re

import operator

import nltk 

import json

from copy import deepcopy

from nltk.tokenize import word_tokenize

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from collections import defaultdict

from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import TfidfVectorizer



from tqdm.notebook import tqdm

from time import sleep



import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import linear_kernel
#Helper Functions by https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv



def format_name(author):

    middle_name = " ".join(author['middle'])

    

    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])





def format_affiliation(affiliation):

    text = []

    location = affiliation.get('location')

    if location:

        text.extend(list(affiliation['location'].values()))

    

    institution = affiliation.get('institution')

    if institution:

        text = [institution] + text

    return ", ".join(text)



def format_authors(authors, with_affiliation=False):

    name_ls = []

    

    for author in authors:

        name = format_name(author)

        if with_affiliation:

            affiliation = format_affiliation(author['affiliation'])

            if affiliation:

                name_ls.append(f"{name} ({affiliation})")

            else:

                name_ls.append(name)

        else:

            name_ls.append(name)

    

    return ", ".join(name_ls)



def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    

    return body



def format_bib(bibs):

    if type(bibs) == dict:

        bibs = list(bibs.values())

    bibs = deepcopy(bibs)

    formatted = []

    

    for bib in bibs:

        bib['authors'] = format_authors(

            bib['authors'], 

            with_affiliation=False

        )

        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]

        formatted.append(", ".join(formatted_ls))



        

    return "; ".join(formatted)





def load_files(dirname):

    filenames = os.listdir(dirname)

    raw_files = []



    for filename in tqdm(filenames):

        filename = dirname + filename

        file = json.load(open(filename, 'rb'))

        raw_files.append(file)

    

    return raw_files



def generate_clean_df(all_files):

    cleaned_files = []

    

    for file in tqdm(all_files):

        features = [

            file['paper_id'],

            file['metadata']['title'],

            format_authors(file['metadata']['authors']),

            format_authors(file['metadata']['authors'], 

                           with_affiliation=True),

            format_body(file['abstract']),

            format_body(file['body_text']),

            format_bib(file['bib_entries']),

            file['metadata']['authors'],

            file['bib_entries']

        ]



        cleaned_files.append(features)



    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df = clean_df.drop(['raw_authors', 'raw_bibliography'], axis=1)

    clean_df.head()

    

    return clean_df
#pmc_dir = '/kaggle/input/CORD-19-research-challenge/pmc_custom_license/pmc_custom_license/'

#pmc_files = load_files(pmc_dir)

#all_df = generate_clean_df(pmc_files)



biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'

biorxiv_files = load_files(biorxiv_dir)

all_df = generate_clean_df(biorxiv_files)



noncomm_dir="/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/"

noncomm_files = load_files(noncomm_dir)

all_df = all_df.append(generate_clean_df(noncomm_files))



comm_dir="/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/"

comm_files = load_files(comm_dir)

all_df = all_df.append(generate_clean_df(comm_files))



print(all_df.shape)

all_df.head()
#i had to do this in iterations so had to make sure that the data is in the correct order

all_df=all_df.sort_values(by=['paper_id'])
#Combining the info so that we can search based on all the given information

all_df['useful_info']=all_df.title+all_df.authors+all_df.affiliations+all_df.abstract+all_df.text+all_df.bibliography
all_df.head()
all_df.useful_info =all_df.useful_info.replace(to_replace='[!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~]',value=' ',regex=True) #remove punctuation

all_df.useful_info =all_df.useful_info.replace(to_replace='-',value=' ',regex=True)

all_df.useful_info =all_df.useful_info.replace(to_replace='\s+',value=' ',regex=True)    #remove new line

all_df.useful_info =all_df.useful_info.replace(to_replace='  ',value='',regex=True)   #remove double white space  

all_df.useful_info =all_df.useful_info.replace(to_replace="'",value='',regex=True)   





all_df.useful_info =all_df.useful_info.apply(lambda x:x.strip())  # Ltrim and Rtrim of whitespace



all_df['useful_info']=[entry.lower() for entry in all_df['useful_info']] #Lowercase
all_df['info_tokenize']= [word_tokenize(entry) for entry in tqdm(all_df.useful_info)] #Tokenize
def wordLemmatizer(data):

    tag_map = defaultdict(lambda : wn.NOUN)

    tag_map['J'] = wn.ADJ

    tag_map['V'] = wn.VERB

    tag_map['R'] = wn.ADV

    clean_k =pd.DataFrame()

    word_Lemmatized = WordNetLemmatizer()

    for index,entry in tqdm(enumerate(data)):

        

        Final_words = []

        for word, tag in pos_tag(entry):

            if len(word)>1 and word not in stopwords.words('english') and word.isalpha():

                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])

                Final_words.append(word_Final)

        

        clean_k.loc[index,'Keyword_final'] = str(Final_words)

    clean_k=clean_k.replace(to_replace ="'", value = '', regex = True)

    clean_k=clean_k.replace(to_replace =" ", value = '', regex = True)

    clean_k=clean_k.replace(to_replace ="\[", value = '', regex = True)

    clean_k=clean_k.replace(to_replace ='\]', value = '', regex = True)

    return clean_k
wordLemmatizer(all_df['info_tokenize'][:10])
del all_df



all_df = pd.read_csv('/kaggle/input/covid19-allresearchpapers-lemmatizedinformation/COVID-19_AllResearchPapers_LemmatizedInformation.csv')
print(all_df.shape)

all_df.head()
#Using Google Universal Sentence Encoder

USEmodel = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

train = USEmodel(all_df.document_keyword)

train_m = tf.train.Checkpoint(v=tf.Variable(train))



train_m.f = tf.function( lambda  x: exported_m.v * x, input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])



model = train_m.v.numpy()
model.shape
def SearchDocument(query,size=10):

    q =[query]

    Q_Train = USEmodel(q)

    

    linear_similarities = linear_kernel(Q_Train, model).flatten() 

    

    Top_index_doc = linear_similarities.argsort()[:-(size+1):-1]

    #print(Top_index_doc)

    linear_similarities.sort()

    find = pd.DataFrame()

    for i,index in enumerate(Top_index_doc):

        find.loc[i,'index'] = str(index)

        find.loc[i,'Paper_ID'] = all_df['paper_id'][index] 

        find.loc[i,'Title'] = all_df['title'][index] 

    for j,simScore in enumerate(linear_similarities[:-(size+1):-1]):

        find.loc[j,'Score'] = simScore

        

    if size==1:

        if find.isnull().values.any():

            print("Query: ",query,".  Title of the Research Paper is missing, Paper ID is:",find.loc[0,'Paper_ID'],"\n")

        else:

            print("Query: ",query,".  Title of the Research Paper:",find.loc[0,'Title'],"\n")

    else:

        return find
SearchDocument("What is Corona Virus")
SearchDocument("What is known about transmission, incubation, and environmental stability?",1)

SearchDocument("Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.",1)

SearchDocument("Prevalence of asymptomatic shedding and transmission (e.g., particularly children).",1)

SearchDocument("Seasonality of transmission.",1)

SearchDocument("Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).",1)

SearchDocument("Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).",1)

SearchDocument("Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).",1)

SearchDocument("Natural history of the virus and shedding of it from an infected person",1)

SearchDocument("Implementation of diagnostics and products to improve clinical processes",1)

SearchDocument("Disease models, including animal models for infection, disease and transmission",1)

SearchDocument("Tools and studies to monitor phenotypic change and potential adaptation of the virus",1)

SearchDocument("Immune response and immunity",1)

SearchDocument("Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",1)

SearchDocument("Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",1)

SearchDocument("Role of the environment in transmission",1)
SearchDocument("What do we know about COVID-19 risk factors?",1)

SearchDocument("Data on potential risks factors",1)

SearchDocument("Smoking, pre-existing pulmonary disease",1)

SearchDocument("Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities",1)

SearchDocument("Neonates and pregnant women",1)

SearchDocument("Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.",1)

SearchDocument("Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors")

SearchDocument("Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups",1)

SearchDocument("Susceptibility of populations",1)

SearchDocument("Public health mitigation measures that could be effective for control",1)

SearchDocument("What do we know about virus genetics, origin, and evolution?",1)

SearchDocument("Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.",1)

SearchDocument("Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.",1)

SearchDocument("Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.",1)

SearchDocument("Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.",1)

SearchDocument("Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.",1)

SearchDocument("Experimental infections to test host range for this pathogen.",1)

SearchDocument("Animal host(s) and any evidence of continued spill-over to humans",1)

SearchDocument("Socioeconomic and behavioral risk factors for this spill-over",1)

SearchDocument("Sustainable risk reduction strategies",1)

SearchDocument("Are there geographic variations in the rate of COVID-19 spread?",1)

SearchDocument("Are there geographic variations in the mortality rate of COVID-19?",1)

SearchDocument("Is there any evidence to suggest geographic based virus mutations?",1)
SearchDocument("What do we know about non-pharmaceutical interventions?",1)

SearchDocument("Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.",1)

SearchDocument("Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.",1)

SearchDocument("Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.",1)

SearchDocument("Methods to control the spread in communities, barriers to compliance and how these vary among different populations.",1)

SearchDocument("Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.",1)

SearchDocument("Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.",1)

SearchDocument("Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).",1)

SearchDocument("Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay.",1)
SearchDocument("What do we know about vaccines and therapeutics?",1)

SearchDocument("Effectiveness of drugs being developed and tried to treat COVID-19 patients.",1)

SearchDocument("Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.",1)

SearchDocument("Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.",1)

SearchDocument("Exploration of use of best animal models and their predictive value for a human vaccine.",1)

SearchDocument("Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.",1)

SearchDocument("Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.",1)

SearchDocument("Efforts targeted at a universal coronavirus vaccine.",1)

SearchDocument("Efforts to develop animal models and standardize challenge studies",1)

SearchDocument("Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers",1)

SearchDocument("Approaches to evaluate risk for enhanced disease after vaccination",1)

SearchDocument("Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]",1)
SearchDocument("What do we know about diagnostics and surveillance?",1)

SearchDocument("How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).",1)

SearchDocument("Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.",1)

SearchDocument("Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.",1)

SearchDocument("National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).",1)

SearchDocument("Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.",1)

SearchDocument("Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).",1)

SearchDocument("Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.",1)

SearchDocument("Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.",1)

SearchDocument("Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.",1)

SearchDocument("Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.",1)

SearchDocument("Policies and protocols for screening and testing.",1)

SearchDocument("Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.",1)

SearchDocument("Technology roadmap for diagnostics.",1)

SearchDocument("Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.",1)

SearchDocument("New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.",1)

SearchDocument("Coupling genomics and diagnostic testing on a large scale.",1)

SearchDocument("Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.",1)

SearchDocument("Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.",1)

SearchDocument("One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.",1)

SearchDocument("What has been published about information sharing and inter-sectoral collaboration?",1)

SearchDocument("Methods for coordinating data-gathering with standardized nomenclature.",1)

SearchDocument("Sharing response information among planners, providers, and others.",1)

SearchDocument("Understanding and mitigating barriers to information-sharing.",1)

SearchDocument("How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).",1)

SearchDocument("Integration of federal/state/local public health surveillance systems.",1)

SearchDocument("Value of investments in baseline public health response infrastructure preparedness",1)

SearchDocument("Modes of communicating with target high-risk populations (elderly, health care workers).",1)

SearchDocument("Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).",1)

SearchDocument("Communication that indicates potential risk of disease to all population groups.",1)

SearchDocument("Misunderstanding around containment and mitigation.",1)

SearchDocument("Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.",1)

SearchDocument("Measures to reach marginalized and disadvantaged populations.",1)

SearchDocument("Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.",1)

SearchDocument("Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.",1)

SearchDocument("Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care",1)
SearchDocument("Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019",1)

SearchDocument("Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight",1)

SearchDocument("Efforts to support sustained education, access, and capacity building in the area of ethics",1)

SearchDocument("Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.",1)

SearchDocument("Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)",1)

SearchDocument("Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.",1)

SearchDocument("Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.",1)
SearchDocument("What has been published about medical care?",1)

SearchDocument("Resources to support skilled nursing facilities and long term care facilities.",1)

SearchDocument("Mobilization of surge medical staff to address shortages in overwhelmed communities",1)

SearchDocument("Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies",1)

SearchDocument("Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients",1)

SearchDocument("Outcomes data for COVID-19 after mechanical ventilation adjusted for age.",1)

SearchDocument("Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.",1)

SearchDocument("Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.",1)

SearchDocument("Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.",1)

SearchDocument("Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.",1)

SearchDocument("Guidance on the simple things people can do at home to take care of sick people and manage disease.",1)

SearchDocument("Oral medications that might potentially work.",1)

SearchDocument("Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually.",1)

SearchDocument("Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcomes.",1)

SearchDocument("Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials",1)

SearchDocument("Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials",1)

SearchDocument("Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)",1)