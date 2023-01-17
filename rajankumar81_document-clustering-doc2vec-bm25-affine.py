### importing the libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import json



import matplotlib.pyplot as plt

plt.style.use('ggplot')
## reading the matadata information

root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

covid_data = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

covid_data.head()
### loading  all the json files

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
## helper class to read the json files

class FileReader:

    def __init__(self,filepath):

        with open(filepath) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            

            # extract abstract

            for i in content['abstract']:

                self.abstract.append(i['text'])

                

            # extract body text

            for i in content['body_text']:

                self.body_text.append(i['text'])

                

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

            

            def __repr__(self):

                return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'



def get_breaks(content, length):

    data = ""

    words = content.split(' ')

    total_chars = 0



    # add break every length characters

    for i in range(len(words)):

        total_chars += len(words[i])

        if total_chars > length:

            data = data + "<br>" + words[i]

            total_chars = 0

        else:

            data = data + " " + words[i]

    return data
### conversion to a dataframe

dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    try:

        content = FileReader(entry)

    except Exception as e:

        continue  # invalid paper format, skip

    

    

    # get metadata information

    meta_data = covid_data.loc[covid_data['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided

        dict_['abstract_summary'].append("Not provided.")

    elif len(content.abstract.split(' ')) > 100:

        # abstract provided is too long for plot, take first 300 words append with ...

        info = content.abstract.split(' ')[:100]

        summary = get_breaks(' '.join(info), 40)

        dict_['abstract_summary'].append(summary + "...")

    else:

        # abstract is short enough

        summary = get_breaks(content.abstract, 40)

        dict_['abstract_summary'].append(summary)

        

    # get metadata information

    meta_data = covid_data.loc[covid_data['sha'] == content.paper_id]

    

    try:

        # if more than one author

        authors = meta_data['authors'].values[0].split(';')

        if len(authors) > 2:

            # more than 2 authors, may be problem when plotting, so take first 2 append with ...

            dict_['authors'].append(". ".join(authors[:2]) + "...")

        else:

            # authors will fit in plot

            dict_['authors'].append(". ".join(authors))

    except Exception as e:

        # if only one author - or Null valie

        dict_['authors'].append(meta_data['authors'].values[0])

    

    # add the title information, add breaks when needed

    try:

        title = get_breaks(meta_data['title'].values[0], 40)

        dict_['title'].append(title)

    # if title was not provided

    except Exception as e:

        dict_['title'].append(meta_data['title'].values[0])

    

    # add the journal information

    dict_['journal'].append(meta_data['journal'].values[0])

    

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])

df_covid.head()
## information of the dataset

df_covid.info()
# one in all approach - removal of special characters



df_covid['title'].replace('[!"#%\()''*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]','',inplace=True,regex=True)

df_covid['abstract'].replace('[!"#%\()''*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]','',inplace=True,regex=True)

df_covid['abstract_summary'].replace('[!"#%\()''*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]','',inplace=True,regex=True)
## conversion to lower case



def lower_case(input_str):

    input_str = input_str.lower()

    return input_str



 



df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))

df_covid['abstract_summary'] = df_covid['abstract_summary'].apply(lambda x: lower_case(x))
import re

df_covid['abstract'] = df_covid['abstract'].map(lambda x: re.sub('[,\n.!?]', '', x))
## importing the nltk stopwords

import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')

sw = stopwords.words('english')
### importing the lemmatization and tokenization function

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize
### function to apply tokenization, remove stopwords and apply lemmatizer 

def process_message(message,stem = False, stop_words = True,lemmi = True):

    words = word_tokenize(message)

    if stop_words:

        sw = stopwords.words('english')

        words = [word for word in words if word not in sw]

    if stem:

        stemmer = PorterStemmer()

        words = [stemmer.stem(word) for word in words]

    if lemmi:

        lemmatizer = WordNetLemmatizer()

        words = [lemmatizer.lemmatize(word) for word in words]

    return words
## aplying the function to apply tokenization, remove stopwords and apply lemmatizer on the abstract column of the dataframe

df_covid['abstract']=df_covid['abstract'].apply(process_message)
### importing the library

import matplotlib.pyplot as plt
df_covid['journal'].value_counts()[:10].plot.barh(figsize = (10,5))
df_covid['title'].value_counts()[:10].plot.barh(figsize = (10,5))
df_covid['authors'].value_counts()[:10].plot.barh(figsize = (10,5))
## importing the word cloud library

import wordcloud

from wordcloud import WordCloud
df_covid['abstract'] = df_covid['abstract'].astype('str')
wordcloud = WordCloud().generate(' '.join(df_covid['abstract']))

plt.figure(figsize=[30,10])

plt.imshow(wordcloud)

plt.axis("off")
# checking the number of CPU core processor used for this process

import multiprocessing

cores = multiprocessing.cpu_count()

print(cores)
## importing necessary modules

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

from gensim.models import Doc2Vec

from sklearn import utils

from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(list(df_covid['abstract']))]
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)

model_dbow.build_vocab([x for x in tqdm(documents)])
#appending all the vectors in a list for training

X=[]

for i in range(36009):

    X.append(model_dbow.docvecs[i])

    print(model_dbow.docvecs[i])
#import the modules

from sklearn.cluster import KMeans

import numpy as np

#create the kmeans object withe vectors created previously

kmeans = KMeans(n_clusters=10, random_state=0).fit(X)



 



#print all the labels

print(kmeans.labels_)
from sklearn import metrics

labels = kmeans.labels_

centroids = kmeans.cluster_centers_



print ("Cluster id labels for inputted data")

print (labels)

print ("Centroids data")

print (centroids)



print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")

print (kmeans.score(X))
## calcualting the silhouette score

silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')



print ("Silhouette_score: ")

print (silhouette_score)
df_covid["clust"] = labels
## Pie chart visualization showing the percentage of articles belonging to a particular cluster

df_covid.groupby("clust")["abstract"].count().plot(kind = "pie", autopct='%.2f', figsize = (10, 10))
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import gensim

def read_corpus(df, column, tokens_only=False):

    """

    Arguments

    ---------

        df: pd.DataFrame

        column: str 

            text column name

        tokens_only: bool

            wether to add tags or not

    """

    for i, line in enumerate(df[column]):



 



        tokens = gensim.parsing.preprocess_string(line)

        if tokens_only:

            yield tokens

        else:

            # For training data, add tags

            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
## resetting the index of the dataframe

df_covid.reset_index(inplace=True, drop=True)
import random

frac_of_articles = 1

train_df  = df_covid.sample(frac=frac_of_articles, random_state=42)

train_corpus = read_corpus(train_df, 'abstract')
# using distributed memory model

model = gensim.models.doc2vec.Doc2Vec(dm=1, vector_size=60, min_count=5, epochs=20, seed=42, workers=6)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
task_1 = """What is known about transmission, incubation, and environmental stability of COVID-19? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?

Range of incubation periods for the disease in humans (and how this varies across age and health status)

and how long individuals are contagious, even after recovery.

Prevalence of asymptomatic shedding and transmission (e.g., particularly children).

Seasonality of transmission.

Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).

Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).

Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).

Natural history of the virus and shedding of it from an infected person

Implementation of diagnostics and products to improve clinical processes

Disease models, including animal models for infection, disease and transmissionTools and studies to monitor phenotypic change and potential adaptation of the virus

Immune response and immunity

Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings

Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings

Role of the environment in transmission."""







task_2 = """What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?

Data on potential risks factors

Smoking, pre-existing pulmonary disease

Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities

Neonates and pregnant women

Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.

Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors

Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups

Susceptibility of populations

Public health mitigation measures that could be effective for control"""







task_3 = """What do we know about virus genetics, origin, and evolution? What do we know about the virus origin and management measures at the human-animal interface?

Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.

Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.

Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.

Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.

Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.

Experimental infections to test host range for this pathogen.

Animal host(s) and any evidence of continued spill-over to humans

Socioeconomic and behavioral risk factors for this spill-over

Sustainable risk reduction strategies"""







task_4 = """What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics?

Effectiveness of drugs being developed and tried to treat COVID-19 patients.

Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.

Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.

Exploration of use of best animal models and their predictive value for a human vaccine.

Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.

Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.

Efforts targeted at a universal coronavirus vaccine.

Efforts to develop animal models and standardize challenge studies

Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers

Approaches to evaluate risk for enhanced disease after vaccinationAssays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]"""







task_5 = """What do we know about the effectiveness of non-pharmaceutical interventions? What is known about equity and barriers to compliance for non-pharmaceutical interventions?

Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.

Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.

Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.

Methods to control the spread in communities, barriers to compliance and how these vary among different populations..

Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.

Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.

Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).

Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay."""







task_6 = """What do we know about diagnostics and surveillance? What has been published concerning systematic, holistic approach to diagnostics (from the public health surveillance perspective to being able to predict clinical outcomes)?

How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).

Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.

Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.

National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.

Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).

Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.

Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.

Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.

Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.

Policies and protocols for screening and testing.Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.

Technology roadmap for diagnostics.

Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.

New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.

Coupling genomics and diagnostic testing on a large scale.

Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.

Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.

One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors."""







task_7 = """What has been published about medical care? What has been published concerning surge capacity and nursing homes? What has been published concerning efforts to inform allocation of scarce resources? What do we know about personal protective equipment? What has been published concerning alternative methods to advise on disease management? What has been published concerning processes of care? What do we know about the clinical characterization and management of the virus?

Resources to support skilled nursing facilities and long term care facilities.

Mobilization of surge medical staff to address shortages in overwhelmed communities

Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies

Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients

Outcomes data for COVID-19 after mechanical ventilation adjusted for age.

Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.

Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.

Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.

Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.

Guidance on the simple things people can do at home to take care of sick people and manage disease.

Oral medications that might potentially work.

Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually.

Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcomes.

Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials

Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials

Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)

"""





task_8 = """What has been published concerning ethical considerations for research? What has been published concerning social sciences at the outbreak response?

Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019

Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight

Efforts to support sustained education, access, and capacity building in the area of ethics

Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.

Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)

Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.

Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.

"""







task_9 = """What has been published about information sharing and inter-sectoral collaboration? What has been published about data standards and nomenclature? What has been published about governmental public health? What do we know about risk communication? What has been published about communicating with high-risk populations? What has been published to clarify community measures? What has been published about equity considerations and problems of inequity?

Methods for coordinating data-gathering with standardized nomenclature.

Sharing response information among planners, providers, and others.

Understanding and mitigating barriers to information-sharing.

How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).

Integration of federal/state/local public health surveillance systems.

Value of investments in baseline public health response infrastructure preparedness

Modes of communicating with target high-risk populations (elderly, health care workers).

Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).

Communication that indicates potential risk of disease to all population groups.

Misunderstanding around containment and mitigation.

Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.

Measures to reach marginalized and disadvantaged populations.

Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.

Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.

Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care

"""
list_of_tasks = [task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9]
def get_doc_vector(doc):

    tokens = gensim.parsing.preprocess_string(doc)

    vector = model.infer_vector(tokens)

    return vector
abstract_vectors = model.docvecs.vectors_docs

array_of_tasks = [get_doc_vector(task) for task in list_of_tasks]
train_df['abstract_vector'] = [vec for vec in abstract_vectors]
train_array = train_df['abstract_vector'].values.tolist()
from sklearn.neighbors import NearestNeighbors

kd_tree = NearestNeighbors(algorithm='kd_tree', leaf_size=30).fit(train_array)
distances, indices = kd_tree.kneighbors(array_of_tasks, n_neighbors=10)
for i, info in enumerate(list_of_tasks):

    print("="*80, f"\n\nTask = {info[:500]}\n", )

    df =  train_df.iloc[indices[i]]

    abstracts = df['abstract']

    titles = df['title']

    dist = distances[i]

    for l in range(len(dist)):

        print(f" Text index = {indices[i][l]} \n Distance = {distances[i][l]} \n Title: {titles.iloc[l]} \n Abstract extract: {abstracts.iloc[l][:200]}\n\n")
## installing the library

!pip install rank_bm25
## importing the library

from rank_bm25 import BM25Okapi

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import preprocess_documents, preprocess_string
train_array = df_covid.abstract.fillna('').apply(preprocess_string)
bm25_index = BM25Okapi(train_array)
def search(search_string, num_results=10):

    search_tokens = preprocess_string(search_string)

    scores = bm25_index.get_scores(search_tokens)

    top_indexes = np.argsort(scores)[::-1][:num_results]

    return top_indexes
indexes = search(task_1)

indexes
df_covid.loc[indexes,['title','body_text','abstract_summary']]
indexes_task2 = search(task_2)

indexes_task2
df_covid.loc[indexes_task2,['title','body_text','abstract_summary']]
indexes_task3 = search(task_3)

indexes_task3
df_covid.loc[indexes_task3,['title','body_text','abstract_summary']]
indexes_task4 = search(task_4)

indexes_task4
df_covid.loc[indexes_task4,['title','body_text','abstract_summary']]