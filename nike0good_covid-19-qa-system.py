# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import json

import re

import scipy as sc

import warnings



import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# Any results you write to the current directory are saved as output.
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
meta_df.info()
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
all_json[0]
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

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
dict_ = {'paper_id': [], 'doi':[], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    

    try:

        content = FileReader(entry)

    except Exception as e:

        continue  # invalid paper format, skip

    

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['abstract'].append(content.abstract)

    dict_['paper_id'].append(content.paper_id)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided

        dict_['abstract_summary'].append("Not provided.")

    elif len(content.abstract.split(' ')) > 100:

        # abstract provided is too long for plot, take first 100 words append with ...

        info = content.abstract.split(' ')[:100]

        summary = get_breaks(' '.join(info), 40)

        dict_['abstract_summary'].append(summary + "...")

    else:

        # abstract is short enough

        summary = get_breaks(content.abstract, 40)

        dict_['abstract_summary'].append(summary)

        

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    

    try:

        # if more than one author

        authors = meta_data['authors'].values[0].split(';')

        if len(authors) > 2:

            # if more than 2 authors, take them all with html tag breaks in between

            dict_['authors'].append(get_breaks('. '.join(authors), 40))

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

    

    # add doi

    dict_['doi'].append(meta_data['doi'].values[0])

    

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'doi', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])

df_covid.head()
df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))  # word count in abstract

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))  # word count in body

df_covid['body_unique_words']=df_covid['body_text'].apply(lambda x:len(set(str(x).split())))  # number of unique words in body

df_covid.head()
df_covid.info()

df_covid['abstract'].describe(include='all')

df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)

df_covid['abstract'].describe(include='all')
df_covid.head()

df_covid.describe()

df = df_covid.sample(10000, random_state=42)

!pip install langdetect
from tqdm import tqdm

from langdetect import detect

from langdetect import DetectorFactory



# set seed

DetectorFactory.seed = 1000000007



# hold label - language

languages = []



# go through each text

for ii in tqdm(range(0,len(df))):

    # split by space into list, take the first x intex, join with space

    text = df.iloc[ii]['body_text'].split(" ")

    

    lang = "en"

    try:

        if len(text) > 50:

            lang = detect(" ".join(text[:50]))

        elif len(text) > 0:

            lang = detect(" ".join(text[:len(text)]))

    # ught... beginning of the document was not in a good format

    except Exception as e:

        all_words = set(text)

        try:

            lang = detect(" ".join(all_words))

        # what!! :( let's see if we can find any text in abstract...

        except Exception as e:

            

            try:

                # let's try to label it through the abstract then

                lang = detect(df.iloc[ii]['abstract_summary'])

            except Exception as e:

                lang = "unknown"

                pass

    

    # get the language    

    languages.append(lang)
from pprint import pprint



languages_dict = {}

for lang in set(languages):

    languages_dict[lang] = languages.count(lang)

    

print("Total: {}\n".format(len(languages)))

pprint(languages_dict)
df['language'] = languages

plt.bar(range(len(languages_dict)), list(languages_dict.values()), align='center')

plt.xticks(range(len(languages_dict)), list(languages_dict.keys()))

plt.title("Distribution of Languages in Dataset")

plt.show()
df = df[df['language'] == 'en'] 

df.info()

df.head()
df.dropna(inplace=True)

df.info()
task1="""What do we know about virus genetics, origin, and evolution? What do we know about the virus origin and management measures at the human-animal interface?



Specifically, we want to know what the literature reports about:



Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.

Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.

Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.

Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.

Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.

Experimental infections to test host range for this pathogen.

Animal host(s) and any evidence of continued spill-over to humans

Socioeconomic and behavioral risk factors for this spill-over

Sustainable risk reduction strategies"""

task2="""What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?



Specifically, we want to know what the literature reports about:



Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.

Prevalence of asymptomatic shedding and transmission (e.g., particularly children).

Seasonality of transmission.

Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).

Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).

Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).

Natural history of the virus and shedding of it from an infected person

Implementation of diagnostics and products to improve clinical processes

Disease models, including animal models for infection, disease and transmission

Tools and studies to monitor phenotypic change and potential adaptation of the virus

Immune response and immunity

Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings

Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings

Role of the environment in transmission"""

task3="""What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?



Specifically, we want to know what the literature reports about:



Data on potential risks factors

Smoking, pre-existing pulmonary disease

Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities

Neonates and pregnant women

Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.

Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors

Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups

Susceptibility of populations

Public health mitigation measures that could be effective for control"""

task4="""What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics?



Specifically, we want to know what the literature reports about:



Effectiveness of drugs being developed and tried to treat COVID-19 patients.

Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.

Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.

Exploration of use of best animal models and their predictive value for a human vaccine.

Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.

Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.

Efforts targeted at a universal coronavirus vaccine.

Efforts to develop animal models and standardize challenge studies

Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers

Approaches to evaluate risk for enhanced disease after vaccination

Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]"""

task5="""What has been published about medical care? What has been published concerning surge capacity and nursing homes? What has been published concerning efforts to inform allocation of scarce resources? What do we know about personal protective equipment? What has been published concerning alternative methods to advise on disease management? What has been published concerning processes of care? What do we know about the clinical characterization and management of the virus?



Specifically, we want to know what the literature reports about:



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

Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)"""

task6="""What do we know about the effectiveness of non-pharmaceutical interventions? What is known about equity and barriers to compliance for non-pharmaceutical interventions?



Specifically, we want to know what the literature reports about:



Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.

Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.

Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.

Methods to control the spread in communities, barriers to compliance and how these vary among different populations..

Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.

Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.

Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).

Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay."""

task7="""What do we know about diagnostics and surveillance? What has been published concerning systematic, holistic approach to diagnostics (from the public health surveillance perspective to being able to predict clinical outcomes)?



Specifically, we want to know what the literature reports about:



How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).

Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.

Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.

National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).

Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.

Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).

Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.

Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.

Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.

Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.

Policies and protocols for screening and testing.

Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.

Technology roadmap for diagnostics.

Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.

New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.

Coupling genomics and diagnostic testing on a large scale.

Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.

Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.

One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors."""

task8="""What has been published about information sharing and inter-sectoral collaboration? What has been published about data standards and nomenclature? What has been published about governmental public health? What do we know about risk communication? What has been published about communicating with high-risk populations? What has been published to clarify community measures? What has been published about equity considerations and problems of inequity?



Specifically, we want to know what the literature reports about:



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

task9="""What has been published concerning ethical considerations for research? What has been published concerning social sciences at the outbreak response?



Specifically, we want to know what the literature reports about:



Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019

Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight

Efforts to support sustained education, access, and capacity building in the area of ethics

Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.

Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)

Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.

Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media."""
task=[task1,task2,task3,task4,task5,task6,task7,task8,task9]

query=[]

for i in range(len(task)):

    task[i]=task[i].split("\n")

    query.append(task[i][4:])

print(query)
print(df.head)
!pip install -U sentence-transformers

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer



warnings.filterwarnings("ignore")

model = SentenceTransformer('bert-base-nli-mean-tokens')
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer



warnings.filterwarnings("ignore")

model = SentenceTransformer('bert-base-nli-mean-tokens')

    
query_embeddings=[]

for i in range(len(query)):

    query_embeddings.append(model.encode(query[i]))
df.reset_index(drop = True, inplace = True)

df['abstract_summary']

#abstract_embeddings = model.encode(df['abstract'])

abstract_summary_embeddings = model.encode(df['abstract_summary'])

scores=[]

for tsk in range(len(task)):

    

    for prob, query_embedding in zip(query[tsk], query_embeddings[tsk]):

        dis = sc.spatial.distance.cdist([query_embedding], abstract_summary_embeddings, "cosine")[0]



        results = zip(range(len(dis)), dis)

        results = sorted(results, key=lambda x: x[1])

        print("Query:", prob)

        print("Answer:" )

        scores.append(1-results[0][1])

        print(df['abstract'][results[0][0]].strip(), "\n(Score: %.4f)" % (1-results[0][1]),"\n")
plt.hist(scores)

plt.title("BERT Scores")

plt.show()

!pip install wordcloud

import wordcloud

from wordcloud import WordCloud

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import en_core_sci_lg

nlp = en_core_sci_lg.load(disable=["tagger", "parser", "ner"])

nlp.max_length = 2000000

import string



import spacy

from spacy.lang.en.stop_words import STOP_WORDS

punctuations = string.punctuation

stopwords = list(STOP_WORDS)

custom_stop_words = [

    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 

    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 

    'al.', 'Elsevier', 'PMC', 'CZI', 'www'

]

for w in [custom_stop_words,punctuations]:

    if w not in stopwords:

        stopwords.append(w)

parser = en_core_sci_lg.load(disable=["tagger", "ner"])

parser.max_length = 7000000



def spacy_tokenizer(sentence):

    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]

    mytokens = " ".join([i for i in mytokens])

    return mytokens

tqdm.pandas()

df["processed_text"] = df["body_text"].progress_apply(spacy_tokenizer)

df.head()
import seaborn as sns

sns.distplot(df['body_word_count'])

sns.distplot(df['abstract_word_count'])

sns.distplot(df['body_unique_words'])





from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(text, maxx_features):

    

    vectorizer = TfidfVectorizer(max_features=maxx_features)

    X = vectorizer.fit_transform(text)

    return X



df.shape


df['processed_text'].values
text = df['processed_text'].values

X = vectorize( df['processed_text'].values, 2 ** 12)

X
from sklearn.decomposition import PCA



pca = PCA(n_components=0.95, random_state=42)

X_reduced= pca.fit_transform(X.toarray())

X_reduced.shape
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

from sklearn import metrics



# run kmeans with many different k

distortions = []

K = range(2, 50)

for k in K:

    print(k)

    k_means = KMeans(n_clusters=k, random_state=42).fit(X_reduced)

    print(k)

    k_means.fit(X_reduced)

    print(k)

    distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
X_line = [K[0], K[-1]]

Y_line = [distortions[0], distortions[-1]]



# Plot the elbow

plt.plot(K, distortions, 'b-')

plt.plot(X_line, Y_line, 'r')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
k = 20

kmeans = KMeans(n_clusters=k, random_state=42)

y_pred = kmeans.fit_predict(X_reduced)

X_reduced
print(X_reduced.shape,X.shape,y_pred.shape)

df.info()
from sklearn.manifold import TSNE



tsne = TSNE(verbose=1, perplexity=100, random_state=42)

X_embedded = tsne.fit_transform(X.toarray())
p=[]

def get_words(sentence):

    words=sentence.split(' ');

    words=[  word.lower() if word!= "-PRON-" else word for word in words ]

    words_list=[i for i in words if i not in stopwords ]

    # words = " ".join([i for i in mytokens])

    return words

for i in df['abstract_summary']:

    print(get_words(i))

    