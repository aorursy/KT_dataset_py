import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob



import os



import kaggle_uploader

from tqdm.auto import tqdm

tqdm.pandas()
kaggle_uploader.__version__
CORES=4
!mkdir output
!ls /kaggle/input/CORD-19-research-challenge/
class COVDoc:

    def __init__(self, filepath: str):

        self.doc_id = None

        self.filepath = filepath

        self.lang = None

        self.file_type = None

        self.paragraph_tokens = []

        self.paragraph_texts = []

        self.processed_paragraphs = []

#        self.processed_spacy_paragraphs = []

        self.processed_nltk_paragraphs = []

        

    #load_text is used to lazy-load the actual text when needed

    def load_text(self):

        with open(self.filepath) as f:

            d = json.load(f)

            for paragraph in d["body_text"]:

                self.paragraph_texts.append(paragraph["text"].lower())

import json



def describe_doc(doc_path):

    with open(doc_path) as f:

        d = json.load(f)

        print(d.keys())

        print(f"number of paragraphs: {len(d['body_text'])}")

        print()

        for idx, paragraph in enumerate(d["body_text"]):

            print()

            print(f"section {idx+1}: title=", end="")

            print(f'{paragraph["section"]}: {len(paragraph["text"])} chars')

import nltk, re, string, collections

from nltk.util import ngrams

from nltk.corpus import stopwords

import spacy
df_metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

df_metadata[df_metadata["has_pmc_xml_parse"] == True].head()
df_metadata.isnull().sum()
mask = df_metadata["sha"].isnull() & df_metadata["pmcid"].isnull()

dfnulls = df_metadata[mask]

dfnulls.shape
df_metadata[df_metadata["full_text_file"].isnull()]
df_metadata.shape
df_metadata["full_text_file"].unique()
def load_docs_(base_path, file_type):

    if not base_path.endswith("/"):

        base_path = base_path + "/"

    loaded_docs = []

    count_pdf = 0

    count_pmc = 0

    file_paths_pdf = glob.glob(base_path+"pdf_json/*.json")

    file_paths_pmc = glob.glob(base_path+"pmc_json/*.json")

    file_names_pdf = [os.path.basename(path) for path in file_paths_pdf]

    for filepath in tqdm(file_paths_pdf):

        filename_sha = os.path.basename(filepath).split(".")[0]

        #print(filename_sha)

        df_sha = df_metadata[df_metadata["sha"] == filename_sha]

        if df_sha.shape[0] > 0:

            has_pmc = df_sha["has_pmc_xml_parse"].to_list()[0]

            if has_pmc:

                count_pmc += 1

                pmc_id = df_sha["pmcid"].to_list()[0]

                filepath = f"{base_path}pmc_json/{pmc_id}.xml.json"

            else:

                count_pdf += 1

        else:

            count_pdf += 1

        doc = COVDoc(filepath)

        doc.file_type = file_type

        loaded_docs.append(doc)

    print(f"loaded {count_pdf} PDF files, {count_pmc} PMC files of type {file_type}")

    return loaded_docs
file_paths_pdf_all = []

file_paths_pmc_all = []

all_docs = []



def load_doc_paths(base_path, file_type):

    if not base_path.endswith("/"):

        base_path = base_path + "/"

    file_paths_pdf = glob.glob(base_path+"pdf_json/*.json")

    file_paths_pmc = glob.glob(base_path+"pmc_json/*.json")

    file_paths_pdf_all.extend(file_paths_pdf)

    file_paths_pmc_all.extend(file_paths_pmc)
medx_basepath = "/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv"

load_doc_paths(medx_basepath, "medx")
comuse_basepath = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/"

load_doc_paths(comuse_basepath, "comuse")
custom_basepath = "/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/"

load_doc_paths(custom_basepath, "custom")

noncom_basepath = "/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/"

load_doc_paths(noncom_basepath, "noncom")
arxiv_basepath = "/kaggle/input/CORD-19-research-challenge/arxiv/arxiv/"

load_doc_paths(arxiv_basepath, "arxiv")
len(file_paths_pdf_all)
len(file_paths_pmc_all)
def find_docs_in_metadata():

    pmc_count = 0

    pmc_missed = 0

    pdf_count = 0

    pdf_missed = 0

    missed = 0

    total = 0

    for idx, row in tqdm(df_metadata.iterrows(), total=df_metadata.shape[0]):

        pmcid = row["pmcid"]

        found_path = None

        if isinstance(pmcid, str):

            for filepath in file_paths_pmc_all:

                if pmcid in filepath:

                    found_path = filepath

                    #print(filepath)

                    pmc_count += 1

                    break

            if found_path is None:

                #print(pmcid)

                pmc_missed += 1

        if found_path is None:

            sha = row["sha"]

            if isinstance(sha, str):

                for filepath in file_paths_pdf_all:

                    #print(sha)

                    if sha in filepath:

                        found_path = filepath

                        #print(filepath)

                        pdf_count += 1

                        break

            if found_path is None:

                pdf_missed += 1

        if found_path is None:

            missed += 1

        else:

            doc = COVDoc(filepath)

            filetype = filepath.split("/")[-3]

            doc.file_type = filetype

            doc.doc_id = row.cord_uid

            all_docs.append(doc)

        total += 1

    print(f"finished: pmc={pmc_count}, pdf={pdf_count}, missed={missed}, pmc_missed={pmc_missed}, pdf_missed={pdf_missed}")

for filepath in file_paths_pmc_all:

    if "PMC2114261" in filepath:

        print(filepath)

find_docs_in_metadata()
#for doc in all_docs:

#    if "PMC2114261" in doc.filepath:

#        print(doc.doc_id)

#        print(doc.filepath)

#        doc.load_text()

#        print(doc.paragraph_texts)
df_metadata[df_metadata["cord_uid"] == "4sw25blb"]
all_docs[0].doc_id
all_docs[0].filepath
#all_docs[0].load_text()
#all_docs[0].paragraph_texts
def show_nltk_bigrams_for(docs):

    tokens = []

    for doc in docs:

        for paragraph in doc.paragraph_texts:

            doc_tokens = nltk.word_tokenize(paragraph)

            tokens.extend(doc_tokens)

    bigrams = ngrams(tokens, 2)

    bigram_freq = collections.Counter(bigrams)

    print(bigram_freq.most_common(100))
snippets_to_delete = [

    "The copyright holder for this preprint",

    "doi: medRxiv preprint",

    "doi: bioRxiv preprint",

    "medRxiv preprint",

    "bioRxiv preprint",

    "cc-by-nc-nd 4.0",

    "cc-by-nd 4.0",

    "cc-by-nc 4.0",

    "cc-by 4.0", 

    "international license", 

    "is made available under a",

    "(which was not peer-reviewed)",

    "the copyright holder for this preprint",

    "who has granted medrxiv a license to display the preprint in perpetuity",

    "author/funder",

    "all rights reserved",

    "no reuse allowed without permission",

    "all authors declare no competing interests",

    "the authors declare no competing interests",

    "no funding supported the project authors",

    "his article is a US Government work",

    "It is not subject to copyright under 17 USC 105 and is also made available for use under a CC0 license",

    "Images were analyzed and processed using ",

    "ImageJ",

    "(http://imagej.nih.gov/ij)", 

    "Adobe Photoshop"

    "CC 2017",

    "All images were assembled in Adobe Illustrator",

]

snippets_to_delete = [snippet.lower() for snippet in snippets_to_delete]
from urllib.parse import urlparse



def extract_urls(text):

    """Return a list of urls from a text string."""

    out = []

    for word in text.split(' '):

        word = word.replace("(", "")

        word = word.replace(")", "")

        #tried with URL-parse library, it got complicated so really just look for HTTP

#        thing = urlparse(word.strip())

#        if thing.scheme:

        if word.startswith("http:/") or word.startswith("https://"):

            out.append(word)

    return out
#testing the function

extract_urls("(https://voice.baidu.com/act/newpneumonia/newpneumonia/?from=osari_pc_1)")
from collections import defaultdict

import hashlib

from langdetect import detect



#get document language. I made another kernel to list these and about 98% are english, so I keep only the English ones

#otherwise, I considered also translating the remaining ones but it seemed quite expensive looking at the cloud pricings for translate

def get_lang(doc):

    try:

        lang = detect(doc)

    except Exception as e: 

        #some documents are broken as in no text, just garbage. langdetect throws an exception for those

        lang = "broken"

    return lang



#replace URLs with "urlX", remove common template strings

def process_docs(docs):

    print(f"starting process_docs for {len(docs)} docs")

    url_counts = collections.Counter()

    with tqdm(total=len(docs)) as pbar:

        for doc in docs:

            #NOTE: here we finally load the actual document content / paragraphs

            doc.load_text()

            total_text = ""

            for paragraph in doc.paragraph_texts:

                processed_paragraph = process_text(paragraph, url_counts)

                del paragraph

                doc.processed_paragraphs.append(processed_paragraph)

                total_text += processed_paragraph

            #need to save memory, this is not be used later (for now) so delete now

            del doc.paragraph_texts

            doc.lang = get_lang(total_text)

            del total_text

            pbar.update(1)

    print("finished process_docs")

    return url_counts

        

def process_text(doc_text, url_counts):

    #sort url so longest is first, otherwise broken URL replacement if one is subset of another

    urls = extract_urls(doc_text)



    moded_text = doc_text.lower()

    urls_to_replace = []

    for url in urls:

        if not url.startswith("http"):

            continue

        urls_to_replace.append(url)

        url_counts[url] += 1

    del urls



    for snippet in snippets_to_delete:

        moded_text_2 = moded_text.replace(snippet, "")

        del moded_text

        moded_text = moded_text_2



    urls_to_replace.sort(key = len, reverse=True)

    for url in urls_to_replace:

        url_hash = hashlib.sha256(url.lower().encode('utf-8')).hexdigest();

        moded_text_2 = moded_text.replace(url, f"URL{url_hash}")

        del moded_text

        moded_text = moded_text_2

    del urls_to_replace

    return moded_text

!pip install pyenchant

!apt install libenchant-dev -y
import enchant



d = enchant.Dict("en_US")
#print("loading spacy nlp")

#nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])

#some things to try if memory is still an issue:

#https://github.com/explosion/spaCy/issues/3618

def spacy_process(docs_to_process, nlp):

    print("starting spacy process")

    #nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])

    processed_doc_tokens = []

    with tqdm(total=len(docs_to_process)) as pbar:

        for doc in docs_to_process:

            for paragraph in doc.processed_paragraphs:

                spacy_paragraph = nlp(paragraph)

                processed_spacy_paragraph = []

                for token in spacy_paragraph:

                    if token.is_stop:

                        continue

                    if token.pos_ == "PUNCT":

                        continue

                    if token.pos_ == "NUM":

                        #tried to insert _NUM_ for numbers but it dominates quite a bit. just enable this if want it

                        #processed_med_tokens.append("_NUM_")

                        continue

                    token_text = token.lemma_.strip()

                    #spacy seems to have some logic to understand for example that anti-viral is a compound word, but it reports different tokens still

                    #maybe combine them somehow later but for now try it this way..

                    if len(token_text) <= 1:

                        continue

                    processed_spacy_paragraph.append(token_text)

                del spacy_paragraph

                del paragraph

                doc.processed_spacy_paragraphs.append(processed_spacy_paragraph)

            del doc.processed_paragraphs

            pbar.update(1)

    del nlp

    return
replacements = {'cells': 'cell', 'cases': 'case', 'used': 'use', 'using': 'use', 'results': 'result', '2modifications': 'modifications','2substitution': 'substitution','2′omethyltransferase': 'omethyltransferase','2′omethyltransferases': 'omethyltransferase','3adjacent': 'adjacent','3blocking': 'blocking','3coordinate': 'coordinate','3processing': 'processing','5coding': 'coding','5sequence': 'sequence','6phosphate': 'phosphate','abiotic': 'biotic','accuraty': 'accuracy','acidemia': 'academia','adapte': 'adapt','adaptor': 'adapter','adenovirus2': 'adenovirus2','adenoviruses': 'adenovirus','advective': 'adjective','aetiological': 'etiological','ageing': 'aging','aliquote': 'aliquot','alltogether': 'altogether','ambiguus': 'ambiguous','ammonis': 'ammonia','anaesthesia': 'anesthesia','anaesthetize': 'anesthetize','analyse': 'analyze','analysed': 'analyzed','analyte': 'analyze','antarctica': 'antarctic','apathogenic': 'pathogenic','artefact': 'artifact','arteritis': 'arthritis','beest': 'best','begining': 'beginning','behaviour': 'behavior','behavioural': 'behavioral','benchmarke': 'benchmark','binominal': 'binomial','biomedicals': 'biomedical','bulletin6': 'bulletin','caesarean': 'cesarean','capitalise': 'capitalize','carboxyl': 'carbonyl','catalyse': 'catalyze','categorisation': 'categorization','categorise': 'categorize','centralised': 'centralized','chaperone': 'chaperon','characterisation': 'characterization','characterise': 'characterize','characterised': 'characterized','checke': 'check','children1': 'children','chimaera': 'chimera','chimaeric': 'chimeric','circos': 'circus','cirrhosus': 'cirrhosis','cohorte': 'cohort','colinear': 'collinear','collisson': 'collision','colonisation': 'colonization','colour': 'color','colourless': 'colorless','coltd': 'cold','comfirmed': 'confirmed','compacta': 'compact','completetly': 'completely','completness': 'completeness','complexe': 'complex','compostion': 'composition','compounde': 'compound','concentrator5': 'concentrator','conceptualise': 'conceptualize','confluency': 'confluence','conjunctival': 'conjunctiva','contraining': 'containing','convertion': 'conversion','coronaviruses': 'coronavirus','corresponde': 'correspond','criterial': 'criteria','crosstalke': 'crosstalk','crosstalks': 'crosstalk','crystalize': 'crystallize','crystallise': 'crystallize','customisable': 'customizable','customise': 'customize','cyano': 'cyan','cysteines': 'cysteine','cytokines': 'cytokine','cytopathogenicity': 'cytopathogenic','cytotox': 'cytotoxin','cytotoxicities': 'cytotoxin','cytotoxicity': 'cytotoxin','cytotoxins': 'cytotoxin','datasets9': 'datasets','defence': 'defense','derivatize': 'derivative','descendent': 'descendant','destabilise': 'destabilize','detectible': 'detectable','detectr': 'detector','diabete': 'diabetes','dialyzed': 'dialyze','diameter6': 'diameter','diarrhoea': 'diarrhea','differece': 'difference','difine': 'define','dimeter': 'diameter','disc1': 'disc','discernable': 'discernible','discretised': 'discretized','discretize': 'discretized','distinguishs': 'distinguish','distrubution': 'distribution','doublestranded': 'doublestrand','dromedarius': 'dromedaries','ebiosciences': 'biosciences','effectived': 'effective','elegans': 'elegant','elimilate': 'eliminate','elongase': 'elongate','emphasise': 'emphasize','endeavour': 'endeavor','england1': 'england','enrichr': 'enrich','enrolment': 'enrollment','ensembl': 'ensemble','enspire': 'inspire','epithelia': 'epithelial','epitopes': 'epitope','equilocal': 'equivocal','esensor': 'sensor','estimaed': 'estimate','estimatedγ': 'estimated','estimatie': 'estimate','euclidiean': 'euclidean','evalulate': 'evaluate','evaporite': 'evaporate','exclusivly': 'exclusively','exportin5': 'exporting','expresss': 'express','factor2': 'factor','fast5': 'fast','favour': 'favor','favourable': 'favorable','favourably': 'favorably','flagellar': 'flagella','fluorescens': 'fluorescent','formalise': 'formalize','frameshifted': 'frameshift','frameshifter': 'frameshift','frameshifting': 'frameshift','frameshifts': 'frameshift','fulfil': 'fulfill','gastropoda': 'gastropod','geneious': 'generous','generalisation': 'generalization','generalise': 'generalize','generalised': 'generalized','genometric': 'geometric','genomics': 'genomic','grida': 'grid','harbour': 'harbor','hepes': 'herpes','heptatitis': 'hepatitis','heterogeneity': 'heterogenity','heterogenous': 'heterogeneous','holliday': 'holiday','homogenous': 'homogeneous','homolog': 'homology','hospitalisation': 'hospitalization','hospitalise': 'hospitalize','hospitalised': 'hospitalized','hybridisation': 'hybridization','hydrolyse': 'hydrolyze','hydrolysing': 'hydrolyzing','hypothesise': 'hypothesize','ifectious': 'infectious','imager': 'image','immunisation': 'immunization','immuno': 'immune','immunoassays': 'immunoassay','immunoblotting': 'immunoblot','imperiale': 'imperial','inadvertantly': 'inadvertently','incease': 'increase','incremente': 'increment','indictor': 'indicator','individuals5': 'individuals','individualĥ': 'individual','industralized': 'industrialized','infec': 'infect','infecte': 'infect','infecteds': 'infected','infection1': 'infection','infection2': 'infection','infections8': 'infections','influenzae': 'influenza','initialise': 'initialize','instal': 'install','instituitional': 'institutional','instututional': 'institutional','interferonγ': 'interferon','interleukin2': 'interleukin','interleukin6': 'interleukin','interleukin8': 'interleukin','internalisation': 'internalization','interspecie': 'interspecies','intinity': 'infinity','isotype': 'isotope','judgement': 'judgment','labeld': 'labeled','labelling': 'labeling','labour': 'labor','leucocyte': 'leukocyte','libarary': 'library','licence': 'license','lindependent': 'independent','localisation': 'localization','localised': 'localized','logisticα': 'logistics','loop1': 'loop','lysates': 'lysate','makino': 'making','marginalise': 'marginalize','mathematica': 'mathematical','maximisation': 'maximization','maximise': 'maximize','mcherry': 'cherry','mclean': 'clean','measurment': 'measurement','medicine4': 'medicine','mega6': 'mega','metagenomes': 'metagenome','methylated': 'methylate','microbiol': 'microbial','minima': 'minimal','minimise': 'minimize','mobilisation': 'mobilization','modeller': 'modeler','modelling': 'modeling','modulatory': 'modulator','moleculare': 'molecular','monocytes': 'monocyte','morbidit': 'morbidity','multinomialq': 'multinomial','multiplexe': 'multiplex','multiplexed': 'multiplex','nanoparticles': 'nanoparticle','naïve': 'naive','neat1': 'neat','neighbour': 'neighbor','neighbourhood': 'neighborhood','neighbouring': 'neighboring','networkx': 'network','neutralisation': 'neutralization','neutralise': 'neutralize','neutraliza': 'neutralize','neutrophils': 'neutrophil','normalisation': 'normalization','normalise': 'normalize','normalised': 'normalized','notationx': 'notation','notationû': 'notation','npopulation': 'population','nucleases': 'nuclease','nucleolin': 'nucleoli','nucleoside': 'nucleotide','nucleosides': 'nucleotides','oesophagus': 'esophagus','offcial': 'official','omethyltransferases': 'omethyltransferase','oppsitely': 'oppositely','optimem': 'optimum','optimisation': 'optimization','optimise': 'optimize','organisation': 'organization','organise': 'organize','overlapa': 'overlap','overrepresente': 'overrepresented','paediatric': 'pediatric','pagel': 'page','parainfluenza3': 'parainfluenza','parameterisation': 'parameterization','parametrisation': 'parameterization','parametrise': 'parametrize','patients6': 'patients','penalise': 'penalize','peneumonia': 'pneumonia','peptidase4': 'peptidase','peroxydase': 'peroxidase','personel': 'personnel','phenylalanin': 'phenylalanine','phospho': 'phosphor','phylogenetically': 'phylogenetic','phylogenetics': 'phylogenetic','physico': 'physics','physicochemical': 'physiochemical','plateaue': 'plateau','pneumoniae': 'pneumonia','polioviruses': 'poliovirus','polymere': 'polymer','populationsṡ': 'populations','popultion': 'population','predition': 'prediction','prioritise': 'prioritize','prisma': 'prism','programme': 'programmer','promotor': 'promoter','prospero': 'prosper','protozoal': 'protozoa','provence': 'province','pselection': 'selection','punctate': 'punctuate','quencher1': 'quencher','quilty': 'guilty','radiograph': 'radiography','randomised': 'randomized','rateμ': 'rate','realisation': 'realization','realise': 'realize','reanalyse': 'reanalyze','recognise': 'recognize','recptor': 'receptor','reduc': 'reduce','refolde': 'refold','regularisation': 'regularization','regulary': 'regularly','remodeler': 'remodel','remodelling': 'remodeling','renumbere': 'renumber','replicase': 'replicate','represen': 'represent','representa': 'represent','reprograme': 'reprogram','reprograming': 'reprogramming','ressources': 'resources','restricta': 'restrict','reteste': 'retest','ribsomal': 'ribosomal','satisfie': 'satisfied','scheme1': 'scheme','scheme2': 'scheme','scientifica': 'scientific','scrutinise': 'scrutinize','sensitisation': 'sensitization','sensitise': 'sensitize','sensitised': 'sensitized','sequela': 'sequel','sequencher': 'sequencer','serie': 'series','signalling': 'signaling','simillar': 'similar','singlestranded': 'singlestrand','specialise': 'specialize','specialised': 'specialized','specically': 'specially','specrometry': 'spectrometry','spektrophotometer': 'spectrophotometer','stabilise': 'stabilize','standard8': 'standard','standardise': 'standardize','standardised': 'standardized','statiscical': 'statistical','statistially': 'statistically','stereotypy': 'stereotype','stimualate': 'stimulate','stirling': 'stirring','strain3': 'strain','striatum': 'stratum','studies9': 'studies','subprocesse': 'subprocess','subsampled': 'subsample','subspecie': 'subspecies','suceptible': 'susceptible','summarise': 'summarize','superpositione': 'superposition','sympatry': 'sympathy','synchronise': 'synchronize','syndrom': 'syndrome','synthesise': 'synthesize','syringae': 'syringe','tetherin': 'tethering','therminator': 'terminator','thresholde': 'threshold','timeω': 'time','tlymphocyte': 'lymphocyte','transduce': 'transducer','transduced': 'transducer','transducer': 'transduce','transfect': 'transfection','transfectants': 'transfectant','transfected': 'transfection','transfecting': 'transfection','transfections': 'transfection','transferases': 'transferase','transferrable': 'transferable','transferrin': 'transferring','transferrins': 'transferring','translocations': 'translocation','transmid': 'transmit','transmsission': 'transmission','traveller': 'traveler','travelling': 'traveling','treshold': 'threshold','tryple': 'triple','tubercolosis': 'tuberculosis','tumour': 'tumor','unappreciate': 'unappreciated','unassemble': 'unassembled','uncoate': 'uncoated','underle': 'underlie','underpowere': 'underpowered','underreporte': 'underreported','undiagnose': 'undiagnosed','unlabelled': 'unlabeled','unpaire': 'unpaired','unrecognised': 'unrecognized','unsupervise': 'unsupervised','upregulated': 'upregulate','upregulates': 'upregulate','upregulations': 'upregulate','urbanisation': 'urbanization','usingñ': 'use','using': 'use', 'utilisation': 'utilization','utilise': 'utilize','vaccinees': 'vaccinee','ventilatory': 'ventilator','viremic': 'viremia','virions': 'virion','virus1': 'virus','viruse': 'virus','viruses3': 'viruses','visualisation': 'visualization','visualise': 'visualize','vitros': 'vitro','wellcome': 'welcome','wildtypes': 'wildtype','µorder': 'order','µslide': 'slide','δpressure': 'pressure', 'studies': 'study'}

"10".isnumeric()
stop_words = set(stopwords.words('english'))

stop_words.update(["et", "al", "fig", "eg", "ie", "2′", "usepackage", "setlength", "also", "may", "figure", "one", "two", "new", "however"])

#stop_words
replace_chars = string.punctuation#.replace("_", "")

translator = str.maketrans('', '', replace_chars)

replace_chars
lemmatizer = nltk.stem.WordNetLemmatizer()

lemmatizer.lemmatize("was")
from nltk.tokenize import sent_tokenize, word_tokenize



def nltk_process(docs):

    print("nltk process..")

    lemmatizer = nltk.stem.WordNetLemmatizer()

    nltk_paragraph_tokens = []

    #count how many times each recognized word occurs

    known_words_e = collections.Counter()

    #count how many times each unrecognized word occurs. good for looking for typos and domain words by frequency

    unknown_words_e = collections.Counter()



    with tqdm(total=len(docs)) as pbar:

        print(".", end="")

        for doc in docs:

            doc_tokens = []

            #go through the previously Spacy preprocessed words

#            for spacy_paragraph in doc.processed_spacy_paragraphs:

            for paragraph in doc.processed_paragraphs:

                processed_nltk_paragraph = []

                paragraph_token = word_tokenize(paragraph)



                for token in paragraph_token:

                    #remove special chars as defined before (the "translator" variable)

                    token = token.translate(translator)

                    token = token.strip()

                    if token in stop_words:

                        #check words here before NLTK "lemmatizes" some of them, such as was->wa or has->ha

                        continue

                    token = lemmatizer.lemmatize(token)

                    #replace using my custom mapping where applicable

                    if token in replacements:

                        token = replacements[token]

                    if token in stop_words:

                        #check stop words here a second time, just to be sure...

                        continue

                    if len(token) <= 1:

                        #drop single letters and empty words

                        continue

                    if token.isnumeric():

                        continue

                    #d is the dictionary defined before, so check if the dictionary knows about it

                    if not d.check(token):

                        unknown_words_e[token] += 1

                    else:

                        known_words_e[token] += 1



                    processed_nltk_paragraph.append(token)

                doc_tokens.append(processed_nltk_paragraph)

            nltk_paragraph_tokens.append(doc_tokens)

            del doc.processed_paragraphs

            #del doc.processed_spacy_paragraphs

            pbar.update(1)

    return nltk_paragraph_tokens, known_words_e, unknown_words_e
nltk.stem.WordNetLemmatizer().lemmatize('children')
# few words I looked up from the first docs I processed:

#anti-203 = anti campaign for proposal 203 for marihuana?

# hbss = sickle hemoglobin ?

# qpcr = Real-time polymerase chain reaction

#impinger = tool for airborne sampling

# trizol = TRIzol is a chemical solution used in the extraction of DNA, RNA, and proteins from cells. ( wikipedia )
def show_top_ngrams(tokens, top_size, *ns):

    for n in ns:

        print()

        print(f"{n}-GRAMS:")

        ng = ngrams(tokens, n)

        ngram_freq = collections.Counter(ng)

        for line in ngram_freq.most_common(top_size):

            print(f"{line[1]}: {line[0]}")

 
#ptfe = Polytetrafluoroethylene

#pvc = Polyvinyl chloride

#skc biosampler = https://skcltd.com/products2/bioaerosol-sampling/biosampler.html
from gensim.models import Phrases

import gensim

import os

import json



#https://stackoverflow.com/questions/53694381/print-bigrams-learned-with-gensim

#https://datascience.stackexchange.com/questions/25524/how-does-phrases-in-gensim-work



def create_gensim_ngram_models(params):

    filetypes, paragraph_lists, filenames, doc_ids = params

    tokens = []

    total_paragraphs = 0

    for paragraph_list in paragraph_lists:

        tokens.extend(paragraph_list)

        total_paragraphs += len(paragraph_list)

        

    print("creating bigram")

    bigram = Phrases(tokens, min_count=5, threshold=100)

    print("creating trigram")

    trigram = Phrases(bigram[tokens], threshold=100)

    print("creating bigram-model")

    bigram_mod = gensim.models.phrases.Phraser(bigram) 

    print("creating trigram-model")

    trigram_mod = gensim.models.phrases.Phraser(trigram)



    print("processing docs")

    for idx, paragraph_list in enumerate(tqdm(paragraph_lists)):

        paragraph_tokens = []

        paragraph_tokens_extended = []

        for paragraph in paragraph_list:

            gensim_tokens = trigram_mod[bigram_mod[paragraph]]

            extended = []

            for gensim_token in gensim_tokens:

                extended.append(gensim_token)

                if "_" in gensim_token:

                    extended.extend(gensim_token.split("_"))

            paragraph_tokens.append(gensim_tokens)

            paragraph_tokens_extended.append(extended)

            del paragraph



        doc_id = doc_ids[idx]

        filetype = filetypes[idx]

        filename = os.path.basename(f"{filenames[idx]}")

        filename = os.path.splitext(f"{filename}")[0]

        

        doc = doc_id+"\n"

        doc_json_list = []

        for paragraph in paragraph_tokens:

            doc += " "+" ".join(paragraph)

            doc_json_list.append({"text": paragraph})

        doc_json = {"doc_id": doc_id, "body_text": doc_json_list}

        

        whole_filename = f'output/whole/{filetype}/{filename}.txt'

        paragraph_filename = f'output/paragraphs/{filetype}/{filename}.json'

        os.makedirs(os.path.dirname(whole_filename), exist_ok=True)

        os.makedirs(os.path.dirname(paragraph_filename), exist_ok=True)

        with open(whole_filename, 'w') as f:

            f.write(f"{doc}\n")

        with open(paragraph_filename, 'w') as f:

            json.dump(doc_json, f, indent=4)

            

        del paragraph_list

        del paragraph_tokens

        del paragraph_tokens_extended

        del doc

        del doc_json

        del doc_json_list

    return #gensim_doc_tokens, gensim_doc_tokens_extended



def no_pool_gensim(filetypes, paragraphs, filenames, doc_ids):

    params = (filetypes, paragraphs, filenames, doc_ids)

    return create_gensim_ngram_models(params)



def pool_gensim(filetypes, paragraphs, filenames, doc_ids):

    pool_data = (filetypes, paragraphs, filenames, doc_ids)

    with Pool(processes=1) as pool:

        result = pool.map(create_gensim_ngram_models, [pool_data])

        #gensim_doc_tokens, gensim_doc_tokens_extended = result[0]

        pool.terminate()

        pool.close()

        pool.join()

    return #gensim_doc_tokens, gensim_doc_tokens_extended



from multiprocessing import Pool

import psutil



def process_map_slice(idx_slice):

    idx = idx_slice[0]

    print(f"processing slice {idx}")

    docs = idx_slice[1]

    print(f"processing slice {idx}: process_docs {len(docs)}")

    url_counts = process_docs(docs)

    docs = [doc for doc in docs if doc.lang == "en"]

    #disabled spacy processing due to memory issues. it was a bit slow too for this purpose

    #print(f"processing slice {idx}: spacy_process")

    #nlp = idx_slice[2]

    #spacy_process(docs, nlp)

    print(f"processing slice {idx}: nltk_process")

    processed_nltk_paragraphs, known_words_e, unknown_words_e = nltk_process(docs)

    doc_ids = [doc.doc_id for doc in docs]

    filepaths = [doc.filepath for doc in docs]

    del docs

    del idx_slice

    print(f"processing slice {idx}: done")

    return (idx, processed_nltk_paragraphs, known_words_e, unknown_words_e, url_counts, doc_ids, filepaths)



def map_reduce(docs_to_process):

    #split the given docs to CORES number of subsets so can spawn a process per core to process the subset

    slice_list = np.array_split(docs_to_process, CORES)

    processed_nltk_paragraphs = []

    #track recognized words for later checking against unrecognized (by dictionary check)

    #helpful for fixing common typos etc

    known_words_e = collections.Counter()

    unknown_words_e = collections.Counter()

    url_counts = collections.Counter()

    doc_ids = []

    filepaths = []

    print("creating pool")

    with Pool(processes=CORES) as pool:

        import gc

        gc.collect()

        idx_slices = []

        for idx, doc_slice in enumerate(slice_list):

            idx_slices.append((idx, doc_slice))

        results = pool.map(process_map_slice, idx_slices)

        pool.terminate()

        pool.close()

        pool.join()

        del pool

        print("pool finished")

        del doc_slice

        del slice_list

        del idx_slices

    results.sort(key=lambda tup: tup[0])

    prev_idx = -1

    print("starting merge")

    #print(psutil.virtual_memory())

    for result in results:

        idx_slice, processed_nltk_paragraphs2, known_words_e_2, unknown_words_e_2, url_counts_2, doc_ids_2, filepaths_2 = result            

        assert idx_slice > prev_idx, f"Prev must be < current: {prev_idx} < {idx_slice} fails"

        prev_idx = idx_slice

        processed_nltk_paragraphs.extend(processed_nltk_paragraphs2)

        known_words_e += known_words_e_2

        unknown_words_e += unknown_words_e_2

        url_counts += url_counts_2

        doc_ids.extend(doc_ids_2)

        filepaths.extend(filepaths_2)

        del processed_nltk_paragraphs2

        del known_words_e_2

        del unknown_words_e_2

        del url_counts_2

        del result

    del results

    #filepaths = [doc.filepath for doc in docs_to_process]

    del docs_to_process

    print("finished map-reduce")

    print(psutil.virtual_memory())

    return processed_nltk_paragraphs, known_words_e, unknown_words_e, url_counts, filepaths, doc_ids

#!cat /kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/0a43046c154d0e521a6c425df215d90f3c62681e.json
import gc



gc.get_threshold()
import random



#some of the article sets contain longer docs, we want to process in parallel so make sure every process gets different lengths by shuffling the list

random.shuffle(all_docs)



#results = map_reduce(all_docs)

#processed_nltk_paragraphs, known_words_e, unknown_words_e, url_counts, all_filenames = results
memory_saving_list = np.array_split(all_docs, 1)

#memory_saving_list = [all_docs]

file_types = [doc.file_type for doc in all_docs]

#doc_ids = [doc.doc_id for doc in all_docs]



processed_nltk_paragraphs = []

known_words_e = collections.Counter()

unknown_words_e = collections.Counter()

url_counts = collections.Counter()

filepaths = []

doc_ids = []

for doc_list in memory_saving_list:

#    doc_list = doc_list[:100]

    processed_nltk_paragraphs_2, known_words_e_2, unknown_words_e_2, url_counts_2, filepaths_2, doc_ids_2 = map_reduce(doc_list)

    processed_nltk_paragraphs.extend(processed_nltk_paragraphs_2)

    known_words_e += known_words_e_2

    unknown_words_e += unknown_words_e_2

    url_counts += url_counts_2

    filepaths.extend(filepaths_2)

    doc_ids.extend(doc_ids_2)

    for doc in doc_list:

        del doc

    del doc_list

    #break



del processed_nltk_paragraphs_2

del known_words_e_2

del unknown_words_e_2

del url_counts_2    



del memory_saving_list

#del all_docs

doc_ids[0]
#processed_nltk_paragraphs[0]
filetypes = file_types
#show_nltk_bigrams_for(all_docs)
unknown_words_e.most_common()[:20]
from operator import itemgetter

#https://stackoverflow.com/questions/17243620/operator-itemgetter-or-lambda/17243726#17243726



unknown_list = list(unknown_words_e.items())

unknown_list.sort(key=itemgetter(1), reverse=True)

unknown_str = ""

for unknown in unknown_list:

    unknown_str += f"{unknown[0]}: {unknown[1]}\n"

with open(f'output/unknown.txt', 'w') as f:

    f.write(unknown_str)
!ls -l output
!df -h .
#unknown_list
def test_remove_threshold(n, counter):

    count_kept = 0

    count_removed = 0

    for word_count in reversed(counter.most_common()):

        if word_count[1] > n:

            #if more than N instances of word, do not remove but check later

            count_kept += 1

            continue

        count_removed += 1

    return count_kept, count_removed

list_kept = []

list_removed = []

for x in range(0, 50):

    count_kept, count_removed = test_remove_threshold(x, unknown_words_e)

    list_kept.append(count_kept)

    list_removed.append(count_removed)

df = pd.DataFrame()

df["kept"] = list_kept

df["removed"] = list_removed

df.plot()

pd.set_option('display.max_rows', 50)

df
words_to_remove = set()

words_to_check = []



#TODO: test threahold values



count = 0

count2 = 0

for word_count in reversed(unknown_words_e.most_common()):

    if word_count[1] > 30:

        #if more than N instances of word, do not remove but check later

        #TODO: smaller threshold for typo check?

        words_to_check.append(word_count)

        count2 += 1

        continue



    words_to_remove.add(word_count[0])    

    count += 1

print(f"selected {count} unknown words for removal")

print(f"kept {count2} unknown words")



def remove_infrequent_known_words():

    count = 0

    count2 = 0

    for word_count in reversed(known_words_e.most_common()):

        if word_count[1] > 2:

            #more than N instances of word, stop removing

            count2 += 1

            continue

        words_to_remove.add(word_count[0])

        count += 1

    print(f"selected {count} known words for removal")

    print(f"kept {count2} known words")





    

remove_infrequent_known_words()

    

words_to_check.sort(key=itemgetter(1), reverse=True)

len(words_to_check)
size_before = 0

size_after = 0

new_list_list = []

for paragraph_list in tqdm(processed_nltk_paragraphs):

    new_list = []

    for paragraph in paragraph_list:

        size_before += len(paragraph)

        new_paragraph = [token for token in paragraph if token not in words_to_remove]

        size_after += len(new_paragraph)

        new_list.append(new_paragraph)

        del paragraph

    new_list_list.append(new_list)



del processed_nltk_paragraphs

processed_nltk_paragraphs = new_list_list

print(f"size before:{size_before}")

print(f"size after: {size_after}")

diff = size_before - size_after

print(f"reduced by: {diff}")

    
!pip install weighted-levenshtein

!pip install python-Levenshtein
from weighted_levenshtein import lev, osa, dam_lev

import Levenshtein



#just to see it works

print(lev('BANANAS', 'BANDANAS'))

print(Levenshtein.distance("BANANAS", "BANDANAS"))
import time



def find_closest_matches(unknown_words, known_words):

    count = 0

    count_diff = 0

    count_short = 0

    count_scored = 0

    file_str = ""

    epoch_time = int(time.time()/60)

    for unknown in tqdm(unknown_words):

        if len(unknown[0]) < 5:

            count_short += 1

            #skip saving words that are shorter than 5 chars

            continue

#            if unknown[1] < 20:

#                #less than 10 instances of word, stop

#                break

        for known in known_words:

            unknown_word = unknown[0]

            known_word = known

            diff = len(unknown_word) - len(known_word)

            diff = abs(diff)

            if diff > 1:

                count_diff += 1

                continue

            count_scored += 1

            score = Levenshtein.distance(unknown_word, known_word)

            if score == 1:

                line = f"{unknown[1]}: '{unknown[0]}': '{known}',"

                file_str += f"{line}\n"

                if count < 50:

                    print(line)

                count += 1

    with open(f'output/closest.txt', 'w') as f:

        f.write(file_str)



    print(f"count={count}, count_diff={count_diff}, count_short={count_short}, count_scored={count_scored}")
find_closest_matches(words_to_check, known_words_e)

#find_closest_matches(unknown_words_e, known_words_e)

!ls -l output
url_counts.most_common(20)
del words_to_check

del words_to_remove

del known_words_e

del unknown_words_e

del all_docs

del unknown_list

del unknown_str

%%time

no_pool_gensim(filetypes, processed_nltk_paragraphs, filepaths, doc_ids)

!ls output/paragraphs/custom_license | head -n 10
!du output

#!head output/paragraphs/custom/00016663c74157a66b4d509d5c4edffd5391bbe0.json
!ls output
!ls output/paragraphs/noncomm_use_subset | wc -l
!ls output/whole/noncomm_use_subset | wc -l
!ls output/paragraphs/biorxiv_medrxiv | wc -l
!ls -l output/whole/custom_license | head -n 10
!ls -l output/paragraphs/comm_use_subset | head -n 10

#!head output/whole/comuse/1a465d982030d8f361dc914ff2defa359fdbe5f9.txt
!apt install zip -y
%%time

#!tar zcf output.tgz output

%%time

!zip -r -q output.zip output
!ls -l output
!ls -l
!mkdir upload_dir

!mv output.zip upload_dir
!ls upload_dir
os.path.abspath("./upload_dir")
import kaggle_uploader



from kaggle_secrets import UserSecretsClient



user_secrets = UserSecretsClient()

api_secret = user_secrets.get_secret("kaggle api key")



kaggle_uploader.resources = []

kaggle_uploader.init_on_kaggle("donkeys", api_secret)

kaggle_uploader.base_path = "./upload_dir"

kaggle_uploader.title = "COVID NLP Preprocessed"

kaggle_uploader.dataset_id = "covid-nlp-preprocess"

kaggle_uploader.user_id = "donkeys"

kaggle_uploader.add_resource("output.zip", "zipped preprocessing data")

kaggle_uploader.update("new version")

import collections



word_count = collections.Counter()

for doc in processed_nltk_paragraphs:

    for paragraph in doc:

        word_count.update(paragraph)

len(word_count)
word_count.most_common(30)
!ls output/paragraphs/biorxiv_medrxiv | head -n 10
!head output/paragraphs/biorxiv_medrxiv/00060fb61742ff60e4e3ba4648c74a34cfe9560d.json
!rm -rf output