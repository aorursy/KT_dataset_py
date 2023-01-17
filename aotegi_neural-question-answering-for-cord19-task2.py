!pip install Whoosh # search engine library
import os.path # pathname manipulations

import codecs # base classes for standard Python codecs, like text encodings (UTF-8,...)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json # JSON file manipulations

from IPython.core.display import display, HTML # object displaying in different formats

from whoosh.index import * # whoosh: full-text indexing and searching

from whoosh.fields import *

from whoosh import qparser

import torch # optimized tensor library for deep learning using GPUs and CPUs

from transformers import BertTokenizer, BertForQuestionAnswering, BasicTokenizer # transformers: large-scale transformer models like BERT, and usage scripts for them

from transformers.data.metrics.squad_metrics import _get_best_indexes
# Define path for CORD-19 dataset and its metadata file

path_dataset = '/kaggle/input/CORD-19-research-challenge/'

path_mdata = path_dataset + 'metadata.csv'



# Select interesting fields from metadata file

fields = ['cord_uid','title', 'publish_time', 'abstract', 'journal','url', 'has_pdf_parse', 'has_pmc_xml_parse', 'pmcid', 'full_text_file', 'sha']



# Extract selected fields from metadata file into dataframe

df_mdata = pd.read_csv(path_mdata, skipinitialspace=True, index_col='cord_uid',usecols=fields)



# WARNING: cord_uid is described as unique, but some of them are repeated. So we keep just the first one

# Repeated cord_uids in v7 of data: 0klupmep, 0z5wacxs, 21htepa1, 21qu87oh, 2maferew, 30duqivi, 3ury4hnv, 

#   4fbr8fx8, 4hlvrfeh, 5ei7iwu0, 5kzx5hgg, 6hdoap81, 6z5f2gz3, 79mzwv1c, 7y8fd521, 8fwa2c24, 940au47y,

#   adygntbe, brz1fn2h, c4u0gxp5, e9pwguwm, eich19nx, hox2xwjg, j7swau26, laq5ze8o, m6q8kbjg, mmls866r,

#   o4r34pff, qhftb6d7, sv9mdgek, vp5358rr, vqbreyna, xjpev4jw

df_mdata = df_mdata.loc[~df_mdata.index.duplicated(keep='first')]



# Sanity check

print("Number of papers loaded from metadata (after filtering out the repeated ones):", len(df_mdata))
# List of COVID-19 synonyms

synonyms = [

    'coronavirus 2019',

    'coronavirus disease 19',

    'cov2',

    'cov-2',

    'covid',

    'ncov 2019',

    '2019ncov',

    '2019-ncov',

    '2019 ncov',

    'novel coronavirus',

    'sarscov2',

    'sars-cov-2',

    'sars cov 2',

    'severe acute respiratory syndrome coronavirus 2',

    'wuhan coronavirus',

    'wuhan pneumonia',

    'wuhan virus'

]



# Create a filter with 'False' values

index_list =  list(df_mdata.index.values) 

filter = pd.Series([False] * len(index_list))

filter.index = index_list

  

# Update the filter using the synonym list, checking if a synonym appears in the title or the abstract of a paper

for s in synonyms:

    # Check if a synonym is in title or abstract

    filter = filter | df_mdata.title.str.lower().str.contains(s) | df_mdata.abstract.str.lower().str.contains(s)



# Filter out papers in metadata dataframe using the above filter

df_mdata = df_mdata[filter]



# Sanity check

print("After filtering, number of papers in metadata related to 'COVID-19':", len(df_mdata))
# Schema definition:

# - id: type ID, unique, stored; cord_uid + "##abs" for abstract, and "##pmc-N" or "##pdf-N" for paragraphs in body text (Nth paragraph)

# - path: type ID, stored; path to the JSON file (only for papers with full text)

# - title: type TEXT processed by StemmingAnalyzer; not stored; title of the paper

# - text: type TEXT processed by StemmingAnalyzer; not stored; content of the abstract section or the paragraph

schema = Schema(id = ID(stored=True,unique=True),

                path = ID(stored=True),

                title = TEXT(analyzer=analysis.StemmingAnalyzer()),

                text = TEXT(analyzer=analysis.StemmingAnalyzer())

               )
# Create an index

if not os.path.exists("index"):

    os.mkdir("index")



ix = create_in("index", schema)
# Add papers to the index, iterating through each row in the metadata dataframe

writer = ix.writer()





not_indexed = []

indexed_sha = []



for ind in df_mdata.index: 

    indexed = False

    

    # If paper has an abstract, index the abstract

    if not pd.isnull(df_mdata.loc[ind,'abstract']):

        if pd.isnull(df_mdata.loc[ind,'title']):

            df_mdata.at[ind,'title'] = ""

        # Add document to the index

        writer.add_document(id=ind+"##abs", title=df_mdata['title'][ind], text=df_mdata['abstract'][ind])

        indexed = True

    

    # If paper has PMC or PDF full text, access its JSON file and index each paragraph separately

    # First check if paper has PMC xml

    if df_mdata['has_pmc_xml_parse'][ind] == True:

        if pd.isnull(df_mdata.loc[ind,'title']):

            df_mdata.at[ind,'title'] = ""

        

        # Find JSON file: path specified in 'full_text_file', file name specidfied in 'pmcid'

        path_json = path_dataset + df_mdata['full_text_file'][ind] + '/' + df_mdata['full_text_file'][ind] + '/pmc_json/' + df_mdata['pmcid'][ind] + '.xml.json'

        with open(path_json, 'r') as j:

            jsondata = json.load(j)

            

            ## Iterate through paragraphs of body_text

            for p, paragraph in enumerate(jsondata['body_text']):  

                # Add document to the index

                writer.add_document(id=ind+"##pmc-" + str(p), path = path_json, title=df_mdata['title'][ind], text=paragraph['text'])

                indexed = True

    

    # As current paper does not have PMC, check if it has JSON PDF

    elif df_mdata['has_pdf_parse'][ind] == True:

        if pd.isnull(df_mdata.loc[ind,'title']):

            df_mdata.at[ind,'title'] = ""

        

        # Find JSON file: path specified in 'full_text_file', file name specidfied in 'sha'

        # There could be more than one reference in 'sha' separated by ;

        shas = df_mdata['sha'][ind].split(';')

        for sha in shas:

            sha = sha.strip()

            # Check if paper with this sha has been indexed already

            if sha not in indexed_sha:

                indexed_sha.append(sha)

                path_json = path_dataset + df_mdata['full_text_file'][ind] + '/' + df_mdata['full_text_file'][ind] + '/pdf_json/' + sha + '.json'

                with open(path_json, 'r') as j:

                    jsondata = json.load(j)

            

                    ## iterate through paragraphs of body_text

                    for p, paragraph in enumerate(jsondata['body_text']):  

                        # Add document to the index

                        writer.add_document(id=ind+"##pdf-" + str(p), path = path_json, title=df_mdata['title'][ind], text=paragraph['text'])

                        indexed = True

    

    if not indexed:

        not_indexed.append(ind)

# Save the added documents

writer.commit()

print("Index successfully created")



# Sanity check

print("Number of documents (abstracts and paragraphs of papers) in the index: ", ix.doc_count())

print("Number of papers not indexed (because they don't have neither the abstract nor full text): ", len(not_indexed))
# Input: Question, dataframe that contains metadata info, maximum number of documents to retrieve

def retrieve_docs(qstring, df_metadata, n_docs):



    # Open the searcher for reading the index. The default BM25F algorithm will be used for scoring

    with ix.searcher() as searcher:

        searcher = ix.searcher()

        

        # Define the query parser ('text' will be the default field to search), and set the input query

        q = qparser.QueryParser("text", ix.schema, group=qparser.OrGroup).parse(qstring)

    

        # Search using the query q, and get the n_docs documents, sorted with the highest-scoring documents first

        results = searcher.search(q, limit=n_docs)

        # results is a list of dictionaries where each dictionary is the stored fields of the document (id, path). 'title' and text' are not stored

    

    # Create columns (id, date, journal, title, text and score) for a new dataframe which will be used to store the results

    ids = []

    dates = []

    journals = []

    titles = []

    texts = []

    scores = [] 

    # Iterate over the retrieved documents to fill in the new dataframe

    for hit in results:

        # Add id to the new dataframe

        ids.append(hit['id'])

        

        # As year, title and text are not stored in the index, they are not returned in results object. They have to be extracted from metadata

        # Get paper id and type of section (abstract, full text paragraph from pmc or pdf)

        pid,sect = hit['id'].split("##") # id examples: 'vho70jcx##pmc-1', a5x5ga60##abs

        

        # Add year to the new dataframe

        if pd.isnull(df_metadata.loc[pid,'publish_time']):

            dates.append("")

        else:

            dates.append(df_metadata['publish_time'][pid])

            

        # Add journal to the new dataframe

        if pd.isnull(df_metadata.loc[pid,'journal']):

            journals.append("unknown journal")

        else:

            journals.append(df_metadata['journal'][pid])

        

        # Add title (with link to the doi, if exists) to the new dataframe 

        if pd.isnull(df_metadata.loc[pid,'url']):

            titles.append(df_metadata['title'][pid])

        else:

            titles.append("<a target=blank href=\"" + df_metadata['url'][pid] + "\">" + df_metadata['title'][pid] + "</a>")

        

        # Add text to the new dataframe

        if sect == 'abs': # get text of the abstract (reading from metadata)

            texts.append(df_metadata['abstract'][pid])

        else: # get text of the paragraph (reading from a JSON file)

            # get pmc or pdf, and the number of paragraph in body full text

            json_type,nsect = sect.split("-") # sect examples: 'pmc-1', 'pdf-5'

    

            # path of the JSON file whether text has been extracted from PMC or PDF

            #if json_type == 'pmc':

            #    path_json = path_dataset + df_metadata['full_text_file'][pid] + '/' + df_metadata['full_text_file'][pid] + '/pmc_json/' + df_metadata['pmcid'][pid] + '.xml.json'    

            #else: 

            #    path_json = path_dataset + df_metadata['full_text_file'][pid] + '/' + df_metadata['full_text_file'][pid] + '/pdf_json/' + df_metadata['sha'][pid] + '.json'

            with open(hit['path'], 'r') as j:

                jsondata = json.load(j)

                texts.append(jsondata['body_text'][int(nsect)]['text'])

        

        # Add score to the new dataframe

        scores.append(hit.score)

    

    # Create a dataframe of results with the columns

    df_results = pd.DataFrame()

    df_results['id'] = ids

    df_results['date'] = dates

    df_results['journal'] = journals

    df_results['title'] = titles

    df_results['text'] = texts

    df_results['score'] = scores

    

    

    return df_results

    # Output: Dataframe where each line is a relevant paragraph, and the columns are the following:

    #         id, date, journal, title, text, score

        
# Input: Question, dataframe that returns the retrieve_docs() function, maximum number of answers to extract, maximum length of the answer 

def extract_answers(qstring, df_results, n_answers, max_answer_len):

    

    # Set tokenizer to lower case the paragraph

    basic_tokenizer = BasicTokenizer(do_lower_case=False)

    

    answers = []

    # Iterate over the paragraphs

    for i, context in enumerate(df_results['text']):

        context = ' '.join(basic_tokenizer.tokenize(context))

        # Add for QuAC

        context += ' CANNOTANSWER'

        # Call a function to extract answers from a paragraph (context)

        answers.append(run_qa(qstring, context, n_answers, max_answer_len))  

        # Remove it from context

        context = context.replace(' CANNOTANSWER', '')

        df_results['text'][i] = context

    # Add answer to the results dataframe

    df_results['qa_answers'] = answers

    

    return df_results

    # Output: Dataframe where each line is a relevant paragraph, and the columns are the following:

    #         id, date, journal, title, text, score, qa_answers.

    #         qa_answers is a dictionary containing 'text' (answer), 'score', 'start_index' and 'end_index' (positions of the answer in the paragraph)
# Load the SciBERT model fine tuned for QA with SQuAD 2.0 and QuAC

tokenizer = BertTokenizer.from_pretrained('/kaggle/input/bertsquadquac/checkpoint-42000/')

basic_tokenizer = BasicTokenizer(do_lower_case=False)

model = BertForQuestionAnswering.from_pretrained('/kaggle/input/bertsquadquac/checkpoint-42000/')
# Input: Question, paragraph, maximum number of answers to extract, maximum length of the answer 

def run_qa(question, context, nbest, max_answer_len):

    #Simple sliding window approach for max context cases

    tokenizer_dict = tokenizer.encode_plus(text=question, text_pair=context, max_length=384, stride=120,

                                           return_overflowing_tokens=True, truncation_strategy='only_second')

    input_ids = [tokenizer_dict['input_ids']]

    input_type_ids = [tokenizer_dict['token_type_ids']]

    

    while 'overflowing_tokens' in tokenizer_dict.keys():

        tokenizer_dict = tokenizer.encode_plus(text=tokenizer.encode(question, add_special_tokens=False), text_pair=tokenizer_dict['overflowing_tokens'], 

                                               max_length=384, stride=120, return_overflowing_tokens=True, truncation_strategy='only_second', 

                                               is_pretokenized=True, pad_to_max_length=True)

        input_ids.append(tokenizer_dict['input_ids'])

        input_type_ids.append(tokenizer_dict['token_type_ids'])    

        

    outputs = model(torch.tensor(input_ids), token_type_ids = torch.tensor(input_type_ids)) 

    answers = []

    

    for i in range(len(input_ids)):

        start_logits, end_logits = [output[i].detach().cpu().tolist() for output in outputs] 

        answers += compute_predictions(start_logits, end_logits, input_ids[i], context.lower(), nbest, max_answer_len)

    

    answers.sort(key = lambda x: x['score'], reverse=True)

    return answers[0:nbest]

    # Output: List of dictionaries containing 'text' (answer), 'score', 'start_index' and 'end_index' (positions of the answer in the paragraph)
# Input: start and end logits for the model, ids, paragraph, maximum number of answers to extract, maximum length of the answer 

def compute_predictions(start_logits, end_logits, input_ids, context, nbest, max_answer_length):

    start_indexes = _get_best_indexes(start_logits, nbest + 10)

    end_indexes = _get_best_indexes(end_logits, nbest + 10)

    answers = []

    for start_index in start_indexes:

        for end_index in end_indexes:

            #Avoid invalid predictions

            answer_len = end_index - start_index + 1

            if end_index < start_index:

                continue

            if max_answer_length < answer_len:

                continue

            text = tokenizer.decode(input_ids[start_index:start_index + answer_len], clean_up_tokenization_spaces=False)

            try:

                original_start_index = context.index(text)

                original_end_index = original_start_index + len(text)

            except:

                #If there is any problem when looking for the answer in the text

                #For example:

                # System says answer in is question

                # Or special tokens in answer [SEP] [PAD]

                continue   

            #When answer contains text and cannotanswer remove the cannotanswer part 

            if text != 'cannotanswer':

                text = text.replace(' cannotanswer', '')

            answer = {'text': text.capitalize(),

                     'score': start_logits[start_index] + end_logits[end_index],

                     'start_index': original_start_index,

                     'end_index': original_end_index}

            answers.append(answer)  

    return answers

    # Output: List of dictionaries containing 'text' (answer), 'score', 'start_index' and 'end_index' (positions of the answer in the paragraph)
tasks = [

    {

        'task': "Task1 - What is known about transmission, incubation, and environmental stability?",

        'questions': [

            "Range of incubation periods for the disease in humans",

            "Range of incubation periods for the disease in humans depending on age",

            "Range of incubation periods for the disease in humans depending on health status",

            "How long individuals are contagious?",

            "Prevalence of asymptomatic shedding and transmission",

            "Prevalence of asymptomatic shedding and transmission in children",

            "Seasonality of transmission",

            "Charge distribution",

            "Adhesion to hydrophilic/phobic surfaces",

            "Environmental survival to inform decontamination efforts for affected areas",

            "Viral shedding",

            "Persistence and stability on nasal discharge",

            "Persistence and stability on sputum",

            "Persistence and stability on urine",

            "Persistence and stability on fecal matter",

            "Persistence and stability on blood",

            "Persistence of virus on surfaces of different materials",

            "Persistence of virus on copper",

            "Persistence of virus on stainless steel",

            "Persistence of virus on plastic",

            "Natural history of the virus",

            "Shedding the virus from an infected person",

            "Implementation of diagnostics to improve clinical processes",

            "Implementation of products to improve clinical processes",

            "Disease models, including animal models for infection, disease and transmission",

            "Tools to monitor phenotypic change and potential adaptation of the virus",

            "Studies to monitor phenotypic change and potential adaptation of the virus",

            "Immune response and immunity",

            "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",

            "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",

            "Role of the environment in transmission"

         ]

    },

    {

        'task': "Task 2 - What do we know about COVID-19 risk factors?",

        'questions': [

            "Which are the main risk factors?",

            "Does smoking increase risk for COVID-19?",

            "Is a pre-existing pulmonary disease a risk factor for COVID-19?",

            "Do co-infections increase risk for COVID-19?",

            "Does a respiratory or viral infection increase risk for COVID-19?",

            "Are neonates at increased risk of COVID-19?",

            "Are pregnant women at increased risk of COVID-19?",

            "Is there any socio-economic factor associated with increased risk for COVID-19?",

            "Is there any behavioral factor associated with increased risk for COVID-19?",

            "What is the basic reproductive number?",

            "What is the incubation period?",

            "What are the modes of transmission?",

            "What are the environmental factors?",

            "Risk of fatality among symptomatic hospitalized patients",

            "Risk of fatality among high-risk patient groups",

            "Susceptibility of populations",

            "Public health mitigation measures that could be effective for control"

        ]

    },

    {

        'task': "Task 3 - What do we know about virus genetics, origin, and evolution?",

        'questions': [

            "Real-time tracking of whole genomes to inform the development of diagnostics",

            "Real-time tracking of whole genomes to inform the development of therapeutics",

            "Real-time tracking of whole genomes to track variations of the virus over time",

            "Mechanism for coordinating the rapid dissemination of whole genomes to inform the development of diagnostics",

            "Mechanism for coordinating the rapid dissemination of whole genomes to inform the development of therapeutics",

            "Mechanism for coordinating the rapid dissemination of whole genomes to track variations of the virus over time",

            "Which geographic and temporal diverse sample sets are accessed to understand geographic differences?",

            "Which geographic and temporal diverse sample sets are accessed to understand genomic differences?",

            "Is there more than one strain in circulation?",

            "Is any multi-lateral agreement leveraged such as the Nagoya Protocol?",

            "Is there evidence that livestock could be infected and serve as a reservoir after the epidemic appears to be over?",

            "Has there been any field surveillance to show that livestock could be infected?",

            "Has there been any genetic sequencing to show that livestock could be infected?",

            "Has there been any receptor binding to show that livestock could be infected?",

            "Is there evidence that farmers are infected?",

            "Is there evidence that farmers could have played a role in the origin?",

            "What are the results of the surveillance of mixed wildlife-livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia?",

            "What are the results of the experimental infections to test host range for this pathogen?",

            "Which are the animal hosts?",

            "Is there evidence of continued spill-over to humans from animals?",

            "Which are the socioeconomic and behavioral risk factors for the spill-over to humans from animals?",

            "Sustainable risk reduction strategies"

        ]

    },

    {

        'task': "Task 4 - What do we know about vaccines and therapeutics?",

        'questions': [

            "What is known about the effectiveness of drugs being developed to treat COVID-19 patients?",

            "What is known about the effectiveness of drugs tried to treat COVID-19 patients?",

            "Show results of clinical and bench trials to investigate less common viral inhibitors against COVID-19",

            "Show results of clinical and bench trials to investigate naproxen against COVID-19",

            "Show results of clinical and bench trials to investigate clarithromycin against COVID-19",

            "Show results of clinical and bench trials to investigate Minocyclinethat against COVID-19",

            "Which are the methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients?",

            "What is known about the use of best animal models and their predictive value for a human vaccine?",

            "Capabilities to discover a therapeutic for the disease",

            "Clinical effectiveness studies to discover therapeutics, to include antiviral agents",

            "Which are the models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics?",

            "Efforts targeted at a universal coronavirus vaccine",

            "Efforts to develop animal models and standardize challenge studies",

            "Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers",

            "Approaches to evaluate risk for enhanced disease after vaccination",

            "Assays to evaluate vaccine immune response",

            "Process development for vaccines, alongside suitable animal models"

        ]

    },

    {

        'task': "Task 5 - What has been published about medical care?",

        'questions': [

            "Resources to support skilled nursing facilities",

            "Resources to support long term care facilities",

            "Mobilization of surge medical staff to address shortages in overwhelmed communities",

            "Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS)",

            "Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) for viral etiologies",

            "What are the outcomes of Extracorporeal membrane oxygenation (ECMO) of COVID-19 patients?",

            "What are the outcomes for COVID-19 after mechanical ventilation adjusted for age?",

            "What is known of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19?",

            "What is known of the frequency, manifestations, and course of cardiomyopathy?",

            "What is known of the frequency, manifestations, and course of cardiac arrest?",

            "Application of regulatory standards (e.g., EUA, CLIA)",

            "Ability to adapt care to crisis standards of care level",

            "Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks",

            "Which are the best telemedicine practices?",

            "Which are the facilitators to expand the telemedicine practices?",

            "Which are the specific actions to expand the telemedicine practices?",

            "Guidance on the simple things people can do at home to take care of sick people and manage disease",

            "Which are the oral medications that might potentially work?",

            "Use of artificial intelligence in real-time health care delivery to evaluate interventions",

            "Use of artificial intelligence in real-time health care delivery to evaluate risk factors",

            "Use of artificial intelligence in real-time health care delivery to evaluate outcomes",

            "Which are the challenges, solutions and technologies in hospital flow and organization?",

            "Which are the challenges, solutions and technologies in workforce protection?",

            "Which are the challenges, solutions and technologies in workforce allocation?",

            "Which are the challenges, solutions and technologies in community-based support resources?",

            "Which are the challenges, solutions and technologies in payment?",

            "Which are the challenges, solutions and technologies in supply chain management to enhance capacity, efficiency, and outcomes?",

            "Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials",

            "What has been done to develop a core clinical outcome set to maximize usability of data across a range of trials?",

            "Can adjunctive or supportive intervention (e.g. steroids, high flow oxygen)  improve the clinical outcomes of infected patients?"

        ]

    },

    {

        'task': "Task 6 - What do we know about non-pharmaceutical interventions?",

        'questions': [

            "Which is the best way to scale up NPIs in a more coordinated way to give us time to enhance our health care delivery system capacity to respond to an increase in cases?",

            "Which is the best way to mobilize resources to geographic areas where critical shortfalls are identified?",

            "Rapid design and execution of experiments to examine and compare NPIs currently being implemented",

            "What is known about the efficacy of school closures?",

            "What is known about the efficacy of travel bans?",

            "What is known about the efficacy of bans on mass gatherings?",

            "What is known about the efficacy of social distancing approaches?",

            "Which are the methods to control the spread in communities?",

            "Models of potential interventions to predict costs and benefits depending on race",

            "Models of potential interventions to predict costs and benefits depending on income",

            "Models of potential interventions to predict costs and benefits depending on disability",

            "Models of potential interventions to predict costs and benefits depending on age",

            "Models of potential interventions to predict costs and benefits depending on geographic location",

            "Models of potential interventions to predict costs and benefits depending on immigration status",

            "Models of potential interventions to predict costs and benefits depending on housing status",

            "Models of potential interventions to predict costs and benefits depending on employment status",

            "Models of potential interventions to predict costs and benefits depending on health insurance status",

            "Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs",

            "Why do people fail to comply with public health advice?",

            "Which is the economic impact of any pandemic?",

            "How can we mitigate risks to critical government services in a pandemic?",

            "Alternatives for food distribution and supplies in a pandemic",

            "Alternatives for household supplies in a pandemic",

            "Alternatives for health diagnoses, treatment, and needed care in a pandemic"

        ]

    },

    {

        'task': "Task 7 - What do we know about diagnostics and surveillance?",

        'questions': [

            "Which are the sampling methods to determine asymptomatic disease?",

            "What can we do for early detection of disease?",

            "Is the use of screening of neutralizing antibodies such as ELISAs valid for early detection of disease?",

            "Which are the existing diagnostic platforms?",

            "Which are the existing surveillance platforms?",

            "Recruitment, support, and coordination of local expertise and capacity",

            "How states might leverage universities and private laboratories for testing purposes?",

            "Which are the best ways for communications to public health officials and the public?",

            "What is the speed, accessibility, and accuracy of a point-of-care test?",

            "What is the speed, accessibility, and accuracy of rapid bed-side tests?",

            "Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity",

            "Separation of assay development issues from instruments",

            "Which is the role of the private sector to help quickly migrate assays?",

            "What has been done to track the evolution of the virus?",

            "Latency issues and when there is sufficient viral load to detect the pathogen",

            "What is needed in terms of biological and environmental sampling?",

            "Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression",

            "Policies and protocols for screening and testing",

            "Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents",

            "Technology roadmap for diagnostics",

            "Which are the barriers to developing and scaling up new diagnostic tests?",

            "How future coalition and accelerator models could provide critical funding for diagnostics?",

            "How future coalition and accelerator models could provide critical funding for opportunities for a streamlined regulatory environment?",

            "New platforms and technology (CRISPR) to improve response times",

            "New platforms and technology to employ more holistic approaches",

            "Coupling genomics and diagnostic testing on a large scale",

            "What is needed for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant?",

            "What is needed for sequencing with advanced analytics for unknown pathogens?",

            "What is needed for distinguishing naturally-occurring pathogens from intentional?",

            "What is known about One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens?"

        ]

    },

    {

        'task': "Task 8 - Help us understand how geography affects virality",

        'questions': [

            "Are there geographic variations in the rate of COVID-19 spread?",

            "Are there geographic variations in the mortality rate of COVID-19?",

            "Is there any evidence to suggest geographic based virus mutations?"

        ]

    },

    {

        'task': "Task 9 - What has been published about ethical and social science considerations?",

        'questions': [

            "Articulate and translate existing ethical principles and standards to salient issues in COVID-2019",

            "Embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight",

            "Support sustained education, access, and capacity building in the area of ethics",

            "Establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences",

            "Develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control",

            "How the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients?",

            "Identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media"

        ]

    },

    {

        'task': "Task 10 - What has been published about information sharing and inter-sectoral collaboration?",

        'questions': [

            "Which are the methods for coordinating data-gathering with standardized nomenclature?",

            "Sharing response information among planners, providers, and others",

            "Understanding and mitigating barriers to information-sharing",

            "How to recruit, support, and coordinate local expertise and capacity relevant to public health emergency response?",

            "Integration of federal/state/local public health surveillance systems",

            "Value of investments in baseline public health response infrastructure preparedness",

            "Modes of communicating with target high-risk populations (elderly, health care workers)",

            "Risk communication and guidelines that are easy to understand and follow",

            "Communication that indicates potential risk of disease to all population groups",

            "Misunderstanding around containment and mitigation",

            "Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment",

            "Measures to reach marginalized and disadvantaged populations",

            "Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities",

            "Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment",

            "Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care"

        ]

    }

]
# Creates the HTML code to show all the answers colored gradually in the paragraph

def color_snippet(text,marks):

    # Set colors for answers

    colors = ['#ffebcc', '#ffc266','#ff9900']

    

    # Create HTML code to show the colored paragraph

    html = '<blockquote>'

    current_mark = 0

    for i,mark in enumerate(marks):

        if current_mark != mark:

            if current_mark != 0:

                html += '</span>'

            if mark > 0:

                html += '<span style="background-color: {}">'.format(colors[mark-1])

            current_mark = mark

        html += text[i]

    if current_mark != 0:

        html += '</span>'

    html += '</blockquote>' 

    return html





# Set number of this task

ntask = 2



# Show title of the task

task_title = tasks[ntask-1]['task']

html = html = "<p><h1>" + task_title + "</h1></p><br>"



# Set input parameters of the functions above

# Maximum number of documents to retrieve

max_n_docs = 20

# Maximum number of answers to extract

max_n_answers = 3

# Maximum answer length

max_answer_length = 30

# Amount of Cannotanswers to declare answers as not suitable

threshold = 17



# Iterate over all the questions in a task and call the functions above

for nq,question in enumerate(tasks[ntask-1]['questions']):

    # Call the function to retrieve relevant paragraphs of papers

    df_ir_results = retrieve_docs(question, df_mdata, max_n_docs)

    # Call the function to extract answers from paragraphs

    df_qa_results = extract_answers(question, df_ir_results, max_n_answers, max_answer_length)



    # Show the question

    html += '<br><p><font color="#683E00"><h2>{}</h2></font>'.format(question)

    

    # Count how many non-null answers are extracted for a question

    n_cannotanswer = 0

    for ind in df_qa_results.index:

        answer = df_qa_results['qa_answers'][ind][0] 

        #Take SQuAD and QuAC cases into account

        if answer['text'] == 'Cannotanswer' or len(answer['text'])==0:

            n_cannotanswer += 1

            

    if n_cannotanswer < threshold:

        # Set maximum number of results to show

        max_n_results = 5

        n_results = 0

        for ind in df_qa_results.index:

            if n_results == max_n_results:

                break

            answers = df_qa_results['qa_answers'][ind]

            # If the first answer is non-null, show the answer

            #if answers[0]['text'] != 'CANNOTANSWER':

            

            if answers[0]['text'] != 'Cannotanswer' and len(answers[0]['text']) != 0:

                answer_string = answers[0]['text']

                html += '<br><b>{}</b> <span style="background-color: #dddddd"> [{}, <i>{}</i>, {}]</span><br>'.format(answer_string, df_qa_results['title'][ind], df_qa_results['journal'][ind], df_qa_results['date'][ind])

            

                # Color the paragraph to highlight the answers

                marks = [0] * len(df_qa_results['text'][ind])

               

                for n_ans, answer in enumerate(answers):

                    if answer['text'] != 'Cannotanswer':

                        level = max_n_answers - n_ans

                        start = answer['start_index']

                        if answer['end_index'] >= len(marks):

                            end = len(marks)-1

                        else:

                            end = answer['end_index']

                       

                        for i in range(start,end):

                            if marks[i] < level:

                                marks[i] = level

                html += color_snippet(df_qa_results['text'][ind], marks)

                n_results += 1        

        html += '<hr>'

    else:

        html += '<br><i>No suitable answers found.</i><br>'

        html += '<hr>'

    



# Display the HTML string that contains all the answers

display(HTML(html))



# Save the HTML code of the answers into a file

if not os.path.exists("html"):

    os.mkdir("html")

html_file = codecs.open("/kaggle/working/html/task" + str(ntask) + ".html","w","utf-8")

html_file.write(html)

html_file.close()