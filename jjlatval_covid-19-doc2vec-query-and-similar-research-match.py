import re

import sys

import spacy



REPLACE_WITH_SPACE = ['\n', '\r', '\t', '\xa0', '/', '|', '\\', '\\n', '\\r', '\\t', '~', '_', '*', '#', '=', ':',

                      '&', '(', ')', '[', ']', '<', '>', '{', '}', '-', '+', '\\xa']

REPLACE_WITH_COMMA = ['、', '､']

REPLACE_WITH_EMPTY_STR = ['"', "'", "`", "´", "^", "¨"]





def clean_str(s):

    """

    A simple string cleaning function for filtering typical noise that the text may have.

    :param s: str

    :return: str

        A cleaned string.

    """

    s = str(s).lower()  # ensure that it is indeed a str + lower



    # CASE 1: Remove Internet noise.

    s = re.sub(r'<.*?>', ' ', s)  # Remove HTML tags.

    s = re.sub(r'http\S+\b', '', s)  # Remove links

    s = re.sub(r'\S*@\S*\s?\b', ' ', s)  # Remove email addresses, even faulty ones and @-mentions such as @mary

    

    # CASE 2: Normalize spacing and listings given noise present in data and remove difficult characters.

    for char in REPLACE_WITH_SPACE:

        s = s.replace(char, ' ')

    for char in REPLACE_WITH_COMMA:

        s = s.replace(char, ',')

    for char in REPLACE_WITH_EMPTY_STR:

        s = s.replace(char, '')



    s = re.sub(' +', ' ', s, re.UNICODE)  # Replace multiple spaces with one space.

    return s.strip()  # Remove leading and trailing whitespaces.





class NLP:



    def __init__(self, lang):

        self.lang = lang

        self.nlp = spacy.load(self.lang)



    def _tokenize(self, s):

        return [token.text for token in self.nlp(s)]



    def normalize_text(self, text):

        text = clean_str(text)

        return self._tokenize(text)



from os import chdir

from os.path import join

import glob

import json



from gensim.models.doc2vec import TaggedDocument

from tqdm import tqdm



ROOT_DIR = '/kaggle/input/CORD-19-research-challenge/'





# Configure what kind of datasets exist

MIN_DOC_LEN = 20

BIORXIV_DATADIR = join(ROOT_DIR, 'biorxiv_medrxiv', 'biorxiv_medrxiv', 'pdf_json')

COMM_USE_DATADIR1 = join(ROOT_DIR, 'comm_use_subset', 'comm_use_subset', 'pdf_json')

COMM_USE_DATADIR2 = join(ROOT_DIR, 'comm_use_subset', 'comm_use_subset', 'pmc_json')

CUSTOM_LICENSE_DATADIR1 = join(ROOT_DIR, 'custom_license', 'custom_license', 'pdf_json')

CUSTOM_LICENSE_DATADIR2 = join(ROOT_DIR, 'custom_license', 'custom_license', 'pmc_json')

NONCOMM_USE_SUBSET_DATADIR1 = join(ROOT_DIR, 'noncomm_use_subset', 'noncomm_use_subset', 'pdf_json')

NONCOMM_USE_SUBSET_DATADIR2 = join(ROOT_DIR, 'noncomm_use_subset', 'noncomm_use_subset', 'pmc_json')



DATA_DIRS = [BIORXIV_DATADIR, COMM_USE_DATADIR1, COMM_USE_DATADIR2, CUSTOM_LICENSE_DATADIR1, CUSTOM_LICENSE_DATADIR2,

             NONCOMM_USE_SUBSET_DATADIR1, NONCOMM_USE_SUBSET_DATADIR2]



BIORXIV_DATASET_NAME = 'biorxiv'

COMM_USE_DATASET1_NAME = 'comm_use_1'

COMM_USE_DATASET2_NAME = 'comm_use_2'

CUSTOM_LICENSE_DATASET1_NAME = 'custom_license_1'

CUSTOM_LICENSE_DATASET2_NAME = 'custom_license_2'

NONCOMM_USE_SUBSET_DATASET1_NAME = 'noncomm_use_subset_1'

NONCOMM_USE_SUBSET_DATASET2_NAME = 'noncomm_use_subset_2'





DATASETS = [BIORXIV_DATASET_NAME, COMM_USE_DATASET1_NAME, COMM_USE_DATASET2_NAME, CUSTOM_LICENSE_DATASET1_NAME,

            CUSTOM_LICENSE_DATASET2_NAME, NONCOMM_USE_SUBSET_DATASET1_NAME, NONCOMM_USE_SUBSET_DATASET2_NAME]

DATASETS_PATHS = {k: v for (k, v) in zip(DATASETS, DATA_DIRS)}



# TODO: terminology is a bit bad here. In reality datasets can consists of document sets which consists of documents,

# TODO i.e. a document set would be the appropriate abstraction level from which documents are derived with their

# TODO own "recipes", i.e. how to construct a document for machine learning purposes.





class BaseDataset:



    def __init__(self, path, name, **kwargs):

        self.path = path

        self.name = name

        self.extension = kwargs.get('extension', '.json')

        self.data_cols = kwargs.get('data_cols', {'metadata': ['title'], 'abstract': ['text']})  # abstract

        self.tag_cols = kwargs.get('tag_cols', {'root': ['paper_id']})



    @staticmethod

    def __get_entry_from_line(line, key, val):

        if key == 'root':

            return line.get(val, None)

        lines = line.get(key, {})

        if type(lines) == list:

            entries = []

            for l in lines:

                entry = l.get(val, None)

                if entry:

                    entries.append(entry)

            return '\n'.join(entries)

        return lines.get(val, None)



    def get_document_from_line(self, line):

        docs = []

        for key, vals in self.data_cols.items():

            for val in vals:

                entry = self.__get_entry_from_line(line, key, val)

                if entry:

                    docs.append(entry)

        return '\n'.join(docs)



    def get_tags_from_line(self, line):

        tags = []

        for key, vals in self.tag_cols.items():

            for val in vals:

                entry = self.__get_entry_from_line(line, key, val)

                if entry:

                    tags.append(entry)

        return tags



    def get_documents_labels(self):

        chdir(self.path)

        for fname in glob.glob('*' + self.extension):

            with open(fname, 'r') as f:

                data = json.load(f)

                doc = self.get_document_from_line(data)

                tags = self.get_tags_from_line(data)

                if tags and len(doc) >= MIN_DOC_LEN:  # Do not even consider too short documents

                    yield doc, tags





def initialize_dataset(name, **kwargs):

    return BaseDataset(path=DATASETS_PATHS[name], name=name, **kwargs)





class Datasets:



    def __init__(self, lang='en', **kwargs):

        """

        Do not use this object for anything else besides training models because normalization of documents

        takes a while.

        TODO: in an actual solution documents should be normalized and those normalized documents should be stored

        TODO: as corpus for fast loading when initiating a training procedure.

        :param lang: str

        :param kwargs: dict

        """

        self.datasets = kwargs.get('datasets', DATASETS)

        self.nlp = NLP(lang)

        self.tagged_docs = []

        self.__init_tagged_docs()



    def __init_tagged_docs(self):

        for dataset_name in self.datasets:

            dataset = initialize_dataset(dataset_name)

            for doc, labels in dataset.get_documents_labels():

                self.tagged_docs.append(TaggedDocument(words=self.nlp.normalize_text(doc), tags=labels))



    def __iter__(self, **kwargs):

        for tagged_document in self.tagged_docs:

            yield tagged_document





class DatasetsSearcher:



    def __init__(self, **kwargs):

        """

        A handy searcher object that only has dataset titles and labels present.

        TODO: this object is slow to load. In an ideal solution this would be done in a database.

        :param lang: str

        :param kwargs: dict

        """

        self.datasets = kwargs.get('datasets', DATASETS)

        self.documents = {}

        self.__init_documents()



    def __init_documents(self):

        print("Loading datasets, this may take a couple of minutes...")

        for dataset_name in self.datasets:

            print("loading dataset {}".format(dataset_name))

            dataset = initialize_dataset(dataset_name, **{'data_cols': {'metadata': ['title']}})

            for doc, labels in tqdm(dataset.get_documents_labels()):

                # In this case labels only have one item and it's always unique.

                self.documents[labels[0]] = doc



    def find_title_by_label(self, label):

        return self.documents.get(label, None)

import logging



from gensim.models import Doc2Vec



MODELS_DIR = '/kaggle/input/covid19-doc2vec/'



# No need to deviate from common good parameters except for:

# epochs - because the dataset is small

# min_count - because the dataset is small



# In some recommender system applications negative ns_exponent values (such as -0.5) may be better; please

# refer to Gensim documentation to find that paper, if you are interested.



DOC2VEC_PARAMS = {

    'dm': 0,

    'hs': 0,

    'negative': 5,

    'ns_exponent': 0.75,

    'sample': 10e-5,

    'dbow_words': 1,

    'workers': -1,

    'vector_size': 300,

    'min_count': 3,

    'window': 5,

    'epochs': 10,

    'alpha': 0.025,

    'min_alpha': 0.0025,

}





class Doc2VecEmbedding:



    def __init__(self, lang='en'):

        self.lang = lang

        self.model = None

        self.nlp = NLP(self.lang)



    def __get_model_fpath(self):

        return join(MODELS_DIR, 'doc2vec_%s.model' % self.lang)



    def load(self):

        self.model = Doc2Vec.load(self.__get_model_fpath())



    def fit(self, **kwargs):

        logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

        logging.root.level = logging.INFO

        datasets = Datasets()

        params = DOC2VEC_PARAMS

        params['documents'] = datasets

        self.model = Doc2Vec(**params)

        self.model.save(self.__get_model_fpath())



    def vectorize(self, text):

        tokens = self.nlp.normalize_text(text)

        return self.model.infer_vector(tokens, steps=256)



    def find_most_similar_docs(self, text, topn=10):

        vector = self.vectorize(text)

        return self.model.docvecs.most_similar(positive=[vector], topn=topn)



# All queries. Note that queries were modified to help improve the results, e.g.

# e.g. by replacing demonstrative pronouns with proper nouns

# "this disease" => "Covid-19 disease"

# "xxx of the virus" => "xxx of the corona virus"

# Note that since the model is trained with only Corona virus research, not mentioning it by name should not be a

# big issue



TRANSMISSION_QUERIES = [

    "Covid-19 incubation periods in humans",

    "Covid-19 incubation periods in humans with respect to age and health status",

    "Prevalence of asymptomatic shedding and transmission",

    "Transmission and children",

    "Physical science of the corona virus",

    "Corona virus charge distribution",

    "Corona virus adhesion to hydrophilic and hydrophobic surfaces"

    "Corona virus environmental survival",

    "Corona virus decontamination of affected areas",

    "Persistence and stability of Corona virus on substrates and sources (nasal discharge, sputum, urine, fecal matter, blood)",

    "Natural history of the Corona virus and shedding from an infected person",

    "Implementation of diagnostics and products to improve clinical processes",

    "Corona virus disease models",

    "Corona virus animal models for infection, disease and transmission",

    "Phenotypic change and adaptation of Corona virus",

    "Immune response and immunity",

    "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",

    "Effectiveness of personal protective equirement (PPE)",

    "Personal protective equipment (PPE) usefulness to reduce risk of transmission in health care and community settings",

    "Role of the environment in transmission"

]



RISK_FACTORS_QUERIES = [

    "Corona virus risk factors smoking",

    "Corona virus risk factors plumonary disease",

    "Corona virus co-infections and other co-mobidities",

    "Neonates and pregnant women",

    "Socio-economic and behavioral factors to understand the conomic impact of the Corona virus",

    "Transmission dynamics of the Corona virus",

    "Corona virus basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors",

    "Severity of Covid-19 disease",

    "Covid-19 fatality among symptomatic hospitalized patients and high-risk patient groups",

    "Corona virus susceptibility of populations",

    "Public health mitigation measures that could be effective for Corona virus control"



]



VIRUS_GENETIC_QUERIES = [

    "Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the Corona virus over time",

    "Geographic distribution and genomic differences of Corona virus",

    "Corona virus strains",

    "Corona virus farmers",

    "Southeast Asia wildlife and livestock Corona virus",

    "Corona virus host range",

    "Corona virus animal hosts and continued spill-over to humans",

    "Socioeconomic and behavioral risk factors of spill-over",

    "Sustainable risk reduction strategies"

]



VACCINES_QUERIES = [

    "Effectiveness of drugs being developed and tried to treat COVID-19 patients",

    "Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication",

    "Exploration of use of best animal models and their predictive value for a human vaccine",

    "Capabilities to discover a therapeutic for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents",

    "Models to aid decision makers in determining how to prioritize and distribute therapeutics when production ramps up",

    "Identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need",

    "Efforts targeted at a universal corona virus vaccine.",

    "Efforts to develop animal models and standardize challenge studies",

    "Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers",

    "Approaches to evaluate risk for enhanced disease after vaccination",

    "Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models"

]



MEDICAL_CARE_QUERIES = [

    "Resources to support skilled nursing facilities and long term care facilities.",

    "Mobilization of surge medical staff to address shortages in overwhelmed communities",

    "Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with other organ failure",

    "Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients",

    "Outcomes data for COVID-19 after mechanical ventilation adjusted for age",

    "COVID-19 extrapulmonary manifestations cardiomyopathy and cardiac arrest",

    "Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level",

    "Encouraging and facilitating the production of elastomeric respirators",

    "Best telemedicine practices, barriers and faciitators",

    "Guidance on the simple things people can do at home to take care of sick people and manage disease",

    "Oral medications that might potentially work",

    "Use of AI in real-time health care delivery to evaluate interventions, risk factors and outcomes",

    "Hospital flow and organization and workforce protection best practices",

    "The natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials",

    "Core clinical outcome set to maximize usability of data across a range of trials",

    "Determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)"

]





NON_PHARMACEUTICAL_QUERIES = [

    "Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases",

    "Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments",

    "Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches",

    "Control the spread in communities",

    "Models to predict costs and benefits that take account factors such as race, income, disability, age, geographic location, immigration status, housing status, employment status and health insurance status."

    "Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.",

    "Why people fail to comply with public health advice",

    "Economic impact of pandemic"

]



DIAGNOSTICS_QUERIES = [

    "How widespread is the current Covid-19 exposure to make immediate policy recommendations on mitigation measures",

    "Sampling methods to determine asymptotic cases, e.g. convalescent samples and early detection of disease such as screening of neutralizing antibodies, ELISAs",

    "Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms",

    "Recruitment, support and coordination of local expertise and capacity",

    "National guidance and guidelines about best practices to states",

    "Development of point of care test (rapid influenza test) and rapid bed-side tests",

    "Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR",

    "Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices",

    "Track down the evolution of the Corona virus (genetic drift or mutations) avoid locking into specific reagents and surveillance/detection schemes",

    "Latency issues and sufficient viral load to detect the pathogen and what is needed in terms of biological and environmental sampling",

    "Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease",

    "Predict severe disease progression",

    "Policies and protocols for screening and testing",

    "Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents",

    "Technology roadmap for Corona virus diagnostics",

    "Scaling up new diagnostic tests, future coalition and accelerator models",

    "New platforms and technology (CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases",

    "Coupling genomics and diagnostic testing on a large scale",

    "Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant",

    "Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional",

    "One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors"

]



ETHICAL_QUERIES = [

    "Ethical principles and standards to salient issues in COVID-2019",

    "Embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight",

    "Support sustained education, access and capacity building in ethics",

    "Establish a team at WHO that will be will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences",

    "Develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control",

    "Identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed",

    "Drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media"



]



INFORMATION_SHARING_QUERIES = [

    "Coordinate data gathering with standardized nomenclature",

    "Share response information among planners and providers",

    "Mitigating barriers to information sharing",

    "Recruit, support and coorinate local (non-Federal) expertise and capacity relevant to public health emergency response",

    "Integration of federal, state, local public health surveillance systems",

    "Value of investments in baseline public health response infrastructure preparedness",

    "Modes of communicating with target high-risk populations (elderly and health care workers)",

    "Risk communication and guidelines",

    "Communication that indicates potential risk of disease to all population groups",

    "Misunderstanding around containment and mitigation",

    "Mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment",

    "Measures to reach marginalized and disadvantaged populations",

    "Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities",

    "Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment",

    "Understanding coverage policies related to testing, treatment and care"

]

class QueryModel:



    def __init__(self, lang='en'):

        self.doc2vec = Doc2VecEmbedding(lang)

        self.doc2vec.load()

        self.searcher = DatasetsSearcher()



    def query(self, query):

        """

        Vectorizes a query through Doc2Vec and uses DatasetSearcher to find the research paper with the title

        :param query: str

        :return: list[list, list, list...]

            A list of results with format: [label, similarity, title]

        """

        res = []

        sims = self.doc2vec.find_most_similar_docs(query)

        for sim in sims:

            label = sim[0]

            res.append([label, sim[1], self.searcher.find_title_by_label(label)])

        return res

QUERIES = [TRANSMISSION_QUERIES, RISK_FACTORS_QUERIES, VIRUS_GENETIC_QUERIES,

           VACCINES_QUERIES, MEDICAL_CARE_QUERIES, NON_PHARMACEUTICAL_QUERIES, DIAGNOSTICS_QUERIES,

           ETHICAL_QUERIES, INFORMATION_SHARING_QUERIES]



QUERY_NAMES = ['transmission queries', 'risk factors queries', 'virus genetic queries', 'vaccines queries',

               'medical care queries', 'non-pharmaceutical queries', 'diagnostics queries', 'ethical queries',

               'information sharing queries']



qm = QueryModel()

for query_list, query_name in zip(QUERIES, QUERY_NAMES):

    print('\n\n' + query_name + '\n\n')

    for query in query_list:

        print_out = "Query: {}".format(query) + '\n\n'

        hits = qm.query(query)

        for res in hits:

            print_out += '{}:; {} {:3f}'.format(res[0], res[2], res[1]) + '\n'

        print(print_out)