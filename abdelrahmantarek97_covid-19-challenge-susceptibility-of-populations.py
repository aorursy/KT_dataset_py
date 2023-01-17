import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # For interactions wit the file system

import json # For using Json Files



# For natural language processing

import spacy

from spacy.matcher import Matcher



# Progress meter

from tqdm import tqdm



nlp = spacy.load("en_core_web_sm")
data_files = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        ifile = os.path.join(dirname, filename)

        if ifile.split(".")[-1] == "json":

            data_files.append(ifile)

            

# Print the number of data files

print("There are {} datafiles".format(len(data_files)))
print(data_files[0])
# open the first file in the data_files list

with open(data_files[1],'r') as sample :

    # parse the json object in the file

    sample = json.load(sample)

    # print the key values

    for key, value in sample.items():

        print(key)
# Furthermore, check what the values to these keys contain



print(sample['metadata'].keys())

print('body_text: ',sample['body_text'][0].keys())

print('bib_entries: ',sample['bib_entries'].keys())

print('ref_entries: ', sample['ref_entries'].keys())
# Initialize title list

titles = []



# for each file, read the file and add a key value pair in the titles list

for file in data_files:

    with open(file,'r') as document:

        document = json.load(document)

        titles.append({document["paper_id"] : document["metadata"]["title"]})

        

print(titles[0])

# Initialize abstract list

abstracts = []



# for each file, read the file and add a key value pair in the abstracts list

for file in data_files:

    

    with open(file,'r')as doc:

        

        document = json.load(doc)

        

        abstract = ''

        

        if('abstract' in document.keys()):

            for item in document['abstract']:

                abstract = abstract + item['text']

        

        abstracts.append({document['paper_id'] : abstract})
# Initialize bodies list

bodies = []



# for each file, read the file and add a key value pair in the bodies list

for file in data_files:

    

    with open(file,'r') as doc:

        

        document = json.load(doc)

        

        body = ''

        

        if('body_text' in document.keys()):

            for item in document['body_text']:

                body = body + item['text']

        

        bodies.append({document['paper_id'] : body})
# will be used to retrieve relevant documents

virus_ref = ['covid-19', 'coronavirus', 'cov-2', 'sars-cov-2', 'sars-cov', 'hcov', '2019-ncov']

# will be used to retrieve documents that include symptoms

symptoms = ['weight loss','chills','shivering','convulsions','deformity','discharge','dizziness','vertigo','fatigue','malaise','asthenia','hypothermia','jaundice','muscle weakness','pyrexia','sweats','swelling','swollen','painful lymph node','weight gain','arrhythmia','bradycardia','chest pain','claudication','palpitations','tachycardia','dry mouth','epistaxis','halitosis','hearing loss','nasal discharge','otalgia','otorrhea','sore throat','toothache','tinnitus','trismus','abdominal pain','fever','bloating','belching','bleeding','blood in stool','melena','hematochezia', 'constipation','diarrhea','dysphagia','dyspepsia','fecal incontinence','flatulence','heartburn','nausea','odynophagia','proctalgia fugax','pyrosis','steatorrhea','vomiting','alopecia','hirsutism','hypertrichosis','abrasion','anasarca','bleeding into the skin','petechia','purpura','ecchymosis and bruising','blister','edema','itching','laceration','rash','urticaria','abnormal posturing','acalculia','agnosia','alexia','amnesia','anomia','anosognosia','aphasia and apraxia','apraxia','ataxia','cataplexy','confusion','dysarthria','dysdiadochokinesia','dysgraphia','hallucination','headache','akinesia','bradykinesia','akathisia','athetosis','ballismus','blepharospasm','chorea','dystonia','fasciculation','muscle cramps','myoclonus','opsoclonus','tic','tremor','flapping tremor','insomnia','loss of consciousness','syncope','neck stiffness','opisthotonus','paralysis and paresis','paresthesia','prosopagnosia','somnolence','abnormal vaginal bleeding','vaginal bleeding in early pregnancy', 'miscarriage','vaginal bleeding in late pregnancy','amenorrhea','infertility','painful intercourse','pelvic pain','vaginal discharge','amaurosis fugax','amaurosis','blurred vision','double vision','exophthalmos','mydriasis','miosis','nystagmus','amusia','anhedonia','anxiety','apathy','confabulation','depression','delusion','euphoria','homicidal ideation','irritability','mania','paranoid ideation','suicidal ideation','apnea','hypopnea','cough','dyspnea','bradypnea','tachypnea','orthopnea','platypnea','trepopnea','hemoptysis','pleuritic chest pain','sputum production','arthralgia','back pain','sciatica','Urologic','dysuria','hematospermia','hematuria','impotence','polyuria','retrograde ejaculation','strangury','urethral discharge','urinary frequency','urinary incontinence','urinary retention']

# will be used to retrieve documents that include information of certain susceptible organs (if needed)

organs = ['mouth','teeth','tongue','salivary glands','parotid glands','submandibular glands','sublingual glands','pharynx','esophagus','stomach','small intestine','duodenum','Jejunum','ileum','large intestine','liver','Gallbladder','mesentery','pancreas','anal canal and anus','blood cells','respiratory system','nasal cavity','pharynx','larynx','trachea','bronchi','lungs','diaphragm','Urinary system','kidneys','Ureter','bladder','Urethra','reproductive organs','ovaries','Fallopian tubes','Uterus','vagina','vulva','clitoris','placenta','testes','epididymis','vas deferens','seminal vesicles','prostate','bulbourethral glands','penis','scrotum','endocrine system','pituitary gland','pineal gland','thyroid gland','parathyroid glands','adrenal glands','pancreas','circulatory system','Heart','patent Foramen ovale','arteries','veins','capillaries','lymphatic system','lymphatic vessel','lymph node','bone marrow','thymus','spleen','tonsils','interstitium','nervous system','brain','cerebrum','cerebral hemispheres','diencephalon','the brainstem','midbrain','pons','medulla oblongata','cerebellum','the spinal cord','the ventricular system','choroid plexus','peripheral nervous system','nerves','cranial nerves','spinal nerves','Ganglia','enteric nervous system','sensory organs','eye','cornea','iris','ciliary body','lens','retina','ear','outer ear','earlobe','eardrum','middle ear','ossicles','inner ear','cochlea','vestibule of the ear','semicircular canals','olfactory epithelium','tongue','taste buds','integumentary system','mammary glands','skin','subcutaneous tissue']

# Will be used to retrieve information about groups of people

higher_terms = ['over', 'above', 'higher', 'older', '>', 'over', 'less']

lower_terms = ['under', 'below', 'fewer', 'younger', '<', 'under', 'more']

# will be used to retrieve documents that mention susceptibility

susceptibility_terms = ['susceptible', 'prone', 'weak', 'receptive', 'risk', 'population', 'susceptibility', 'resistance']

# susceptibility may be related to a certain region, hence why this can also be used

continents = ['continent','europe', 'africa', 'north america', 'south america', 'australia', 'asia', 'north pole', 'south pole']

countries = ['country','afghanistan', 'aland islands', 'albania', 'algeria', 'american samoa', 'andorra', 'angola', 'anguilla', 'antarctica', 'antigua and barbuda', 'argentina', 'armenia', 'aruba', 'australia', 'austria', 'azerbaijan', 'bahamas (the)', 'bahrain', 'bangladesh', 'barbados', 'belarus', 'belgium', 'belize', 'benin', 'bermuda', 'bhutan', 'bolivia (plurinational state of)', 'bonaire, sint eustatius and saba', 'bosnia and herzegovina', 'botswana', 'bouvet island', 'brazil', 'british indian ocean territory (the)', 'brunei darussalam', 'bulgaria', 'burkina faso', 'burundi', 'cabo verde', 'cambodia', 'cameroon', 'canada', 'cayman islands (the)', 'central african republic (the)', 'chad', 'chile', 'china', 'christmas island', 'cocos (keeling) islands (the)', 'colombia', 'comoros (the)', 'congo (the democratic republic of the)', 'congo (the)', 'cook islands (the)', 'costa rica', "cote d'ivoire", 'croatia', 'cuba', 'curacao', 'cyprus', 'czechia', 'denmark', 'djibouti', 'dominica', 'dominican republic (the)', 'ecuador', 'egypt', 'el salvador', 'equatorial guinea', 'eritrea', 'estonia', 'ethiopia', 'falkland islands (the) [malvinas]', 'faroe islands (the)', 'fiji', 'finland', 'france', 'french guiana', 'french polynesia', 'french southern territories (the)', 'gabon', 'gambia (the)', 'georgia', 'germany', 'ghana', 'gibraltar', 'greece', 'greenland', 'grenada', 'guadeloupe', 'guam', 'guatemala', 'guernsey', 'guinea', 'guinea-bissau', 'guyana', 'haiti', 'heard island and mcdonald islands', 'holy see (the)', 'honduras', 'hong kong', 'hungary', 'iceland', 'india', 'indonesia', 'iran (islamic republic of)', 'iraq', 'ireland', 'isle of man', 'israel', 'italy', 'jamaica', 'japan', 'jersey', 'jordan', 'kazakhstan', 'kenya', 'kiribati', "korea (the democratic people's republic of)", 'korea (the republic of)', 'kuwait', 'kyrgyzstan', "lao people's democratic republic (the)", 'latvia', 'lebanon', 'lesotho', 'liberia', 'libya', 'liechtenstein', 'lithuania', 'luxembourg', 'macao', 'macedonia (the former yugoslav republic of)', 'madagascar', 'malawi', 'malaysia', 'maldives', 'mali', 'malta', 'marshall islands (the)', 'martinique', 'mauritania', 'mauritius', 'mayotte', 'mexico', 'micronesia (federated states of)', 'moldova (the republic of)', 'monaco', 'mongolia', 'montenegro', 'montserrat', 'morocco', 'mozambique', 'myanmar', 'namibia', 'nauru', 'nepal', 'netherlands (the)', 'new caledonia', 'new zealand', 'nicaragua', 'niger (the)', 'nigeria', 'niue', 'norfolk island', 'northern mariana islands (the)', 'norway', 'oman', 'pakistan', 'palau', 'palestine, state of', 'panama', 'papua new guinea', 'paraguay', 'peru', 'philippines (the)', 'pitcairn', 'poland', 'portugal', 'puerto rico', 'qatar', 'reunion', 'romania', 'russian federation (the)', 'rwanda', 'saint barthelemy', 'saint helena, ascension and tristan da cunha', 'saint kitts and nevis', 'saint lucia', 'saint martin (french part)', 'saint pierre and miquelon', 'saint vincent and the grenadines', 'samoa', 'san marino', 'sao tome and principe', 'saudi arabia', 'senegal', 'serbia', 'seychelles', 'sierra leone', 'singapore', 'sint maarten (dutch part)', 'slovakia', 'slovenia', 'solomon islands', 'somalia', 'south africa', 'south georgia and the south sandwich islands', 'south sudan', 'spain', 'sri lanka', 'sudan (the)', 'suriname', 'svalbard and jan mayen', 'swaziland', 'sweden', 'switzerland', 'syrian arab republic', 'taiwan (province of china)', 'tajikistan', 'tanzania, united republic of', 'thailand', 'timor-leste', 'togo', 'tokelau', 'tonga', 'trinidad and tobago', 'tunisia', 'turkey', 'turkmenistan', 'turks and caicos islands (the)', 'tuvalu', 'uganda', 'ukraine', 'united arab emirates (the)', 'united kingdom of great britain and northern ireland (the)', 'united states minor outlying islands (the)', 'united states of america (the)', 'uruguay', 'uzbekistan', 'vanuatu', 'venezuela (bolivarian republic of)', 'viet nam', 'virgin islands (british)', 'virgin islands (u.s.)', 'wallis and futuna', 'western sahara*', 'yemen', 'zambia', 'zimbabwe']
matchers = {    

    # A matcher that matches with one term or string

    "Term Matcher": lambda term: [{'LOWER': t} for t in term.split(' ')],

    # A matcher that matches with more than one term

    "Terms Matcher": lambda terms: [{"LOWER": {"IN": terms } }],

    # A matcher that matches number suffixes

    "Number Suffix Matcher": lambda periods: [

        {'LIKE_NUM': True},

        {"TEXT": {"REGEX": f'({"|".join(periods)})'}}

    ],

    # A matcher that matches number intervals

    "Number Interval Matcher": lambda periods: [

        {'POS': 'NUM',},

        {'TEXT': {'REGEX': f'({"|".join(periods)})'}, 'OP': '?'},

        {'DEP': 'quantmod', 'OP': '?'},

        {'DEP': 'punct', 'OP': '?'},

        {'DEP': 'prep', 'OP': '?'},

        {'POS': 'NUM'},

        {'TEXT': {'REGEX': f'({"|".join(periods)})'}},

    ],

    # a matcher that matches intervals that describe groups

    "Group Matcher": [

        {"TEXT": {"IN": higher_terms+lower_terms }}

    ]

}
virus_rule = matchers['Terms Matcher'](virus_ref)

# Create a new matcher

matcher = Matcher(nlp.vocab)

matcher.add("virus_matcher", None, virus_rule)



# A dict that holds the number of matches as as a score for each document

docs_virus = {}



# loop over abstracts and get the relevant documents with their scores

for i in tqdm(range(len(abstracts)), position=0) :

    abstract = abstracts[i]

    for paper_id in abstract.keys() :

        doc = nlp(abstract[paper_id])

        matches = matcher(doc)

        number_of_matches = len(matches)

        if number_of_matches > 0 :

            docs_virus[paper_id] = {"abstract" : abstract, "score" : 0}

            if(docs_virus.get(paper_id) is None):

                docs_virus[paper_id]["score"] = number_of_matches

            else :

                docs_virus[paper_id]["score"] += number_of_matches

        
# get the keys of the docs that matched virus terms

keys = list(docs_virus.keys())





susceptibility_rule = matchers['Terms Matcher'](susceptibility_terms)

# Create a new matcher

matcher = Matcher(nlp.vocab)

matcher.add("susceptibility_matcher", None, susceptibility_rule)



# A dict that holds the number of matches as as a score for each document

docs_susceptibility = {}



# loop over abstracts and get the relevant documents with their scores

for i in tqdm(range(len(keys)), position=0) :

    paper_id = keys[i]

    abstract = docs_virus[paper_id]["abstract"][paper_id]

    doc = nlp(abstract)

    matches = matcher(doc)

    number_of_matches = len(matches)

    if number_of_matches > 0 :

        docs_susceptibility[paper_id] = {"abstract" : abstract, "score-1" : docs_virus[paper_id]["score"], "score-2" : 0}

        if(docs_susceptibility.get(paper_id) is None):

            docs_susceptibility[paper_id]["score-2"] = number_of_matches

        else :

            docs_susceptibility[paper_id]["score-2"] += number_of_matches

        
print(len(docs_virus.keys()))

print(len(docs_susceptibility.keys()))
docs_filtered = {}



keys = list(docs_susceptibility.keys())



for i in tqdm(range(len(keys))):

    paper_id = keys[i]

    score = docs_susceptibility[paper_id]["score-1"] + docs_susceptibility[paper_id]["score-2"]

    docs_susceptibility[paper_id]["score"] = score

    if(score >= 7):

        docs_filtered[paper_id] = docs_susceptibility[paper_id]
# Get the size of the filtered keys

keys = list(docs_filtered.keys())

print(len(keys))



# Display an example entry

paper_id = keys[0]

print(docs_filtered[paper_id])
keys = list(docs_filtered.keys())



for i in tqdm(range(len(keys)), position = 0):

    paper_id = keys[i]

    for title in titles:

        

        if len(title) == 0 :

            docs_filtered[paper_id]["title"] = NA

            continue

            

        found = False

        for paper_idT in title.keys():

            if(paper_id == paper_idT):

                found = True

                docs_filtered[paper_id]["title"] = title[paper_idT]

        

        if(found == False):

            continue

        

        break
# Rule for matching virus terms

title_rule1 = matchers['Terms Matcher'](virus_ref)

# Rule for matching susceptibility terms

title_rule2 = matchers['Terms Matcher'](susceptibility_terms)

# Create a matcher for each rule

matcher1 = Matcher(nlp.vocab)

matcher1.add("virus_matcher", None, title_rule1)

matcher2 = Matcher(nlp.vocab)

matcher2.add("susceptibility", None, title_rule2)



keys = list(docs_filtered.keys())



# for each document, get the results of both matchers, and calculate a weighted score

for i in tqdm(range(len(keys)), position = 0):

    paper_id = keys[i]

    title = docs_filtered[paper_id]["title"]

    doc = nlp(title)

    matches1 = matcher1(doc)

    matches2 = matcher2(doc)

    number_of_matches1 = len(matches1)

    number_of_matches2 = len(matches2)

    

    if(number_of_matches1 > 0 ):

        if(number_of_matches2 > 0):

            title_score = (number_of_matches1 + number_of_matches2) * 10

        else : 

            title_score = (number_of_matches1)

    else :       

        title_score = (number_of_matches2)

            

    docs_filtered[paper_id]["score-title"] = title_score
keys = list(docs_filtered.keys())

for i in tqdm(range(len(keys)), position = 0):

    paper_id = keys[i]

    docs_filtered[paper_id]["score"] = docs_filtered[paper_id]["score-1"] + docs_filtered[paper_id]["score-2"] + docs_filtered[paper_id]["score-title"]
with open('/kaggle/working/filtered_docs_scored.json', 'w') as outfile:

    json.dump(docs_filtered, outfile)

    outfile.close()

    

with open('/kaggle/working/virus_docs.json', 'w') as outfile:

    json.dump(docs_virus, outfile)

    outfile.close()
docs_list = []



keys = list(docs_filtered.keys())



for i in tqdm(range(len(keys)), position = 0):

    paper_id = keys[i]

    docs_filtered[paper_id]["paper_id"] = paper_id

    docs_list.append(docs_filtered[paper_id])





sorted_docs = sorted(docs_list, key=lambda d: d['score'], reverse=True)
print("Rank 1 Document of ID " + sorted_docs[0]["paper_id"] + " Title : \n" + sorted_docs[0]["title"] + "\n Score : " + str(sorted_docs[0]["score"]) + "\n" + "\n")

print("Rank 2 Document of ID " + sorted_docs[1]["paper_id"] + " Title : \n" + sorted_docs[1]["title"] + "\n Score : " + str(sorted_docs[1]["score"]) + "\n" + "\n")

print("Rank 3 Document of ID " + sorted_docs[2]["paper_id"] + " Title : \n" + sorted_docs[2]["title"] + "\n Score : " + str(sorted_docs[2]["score"]) + "\n" + "\n")

print("Rank 4 Document of ID " + sorted_docs[3]["paper_id"] + " Title : \n" + sorted_docs[3]["title"] + "\n Score : " + str(sorted_docs[3]["score"]) + "\n" + "\n")

print("Rank 5 Document of ID " + sorted_docs[4]["paper_id"] + " Title : \n" + sorted_docs[4]["title"] + "\n Score : " + str(sorted_docs[4]["score"]) + "\n" + "\n")

print("Rank 6 Document of ID " + sorted_docs[5]["paper_id"] + " Title : \n" + sorted_docs[5]["title"] + "\n Score : " + str(sorted_docs[5]["score"]) + "\n" + "\n")

with open('/kaggle/working/results.json', 'w') as outfile:

    json.dump({"results" : sorted_docs}, outfile)

    outfile.close()