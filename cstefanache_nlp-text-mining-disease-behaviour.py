import os

import re

import json

import math



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns



import spacy

from spacy.matcher import Matcher



from tqdm import tqdm



nlp = spacy.load("en_core_web_sm")
debug = False

articles = {}

stat = { }

for dirpath, subdirs, files in os.walk('/kaggle/input'):

    for x in files:

        if x.endswith(".json"):

            articles[x] = os.path.join(dirpath, x)        

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

virus_ref = ['covid-19', 'coronavirus', 'cov-2', 'sars-cov-2', 'sars-cov', 'hcov', '2019-ncov']

symptoms = ['weight loss','chills','shivering','convulsions','deformity','discharge','dizziness','vertigo','fatigue','malaise','asthenia','hypothermia','jaundice','muscle weakness','pyrexia','sweats','swelling','swollen','painful lymph node','weight gain','arrhythmia','bradycardia','chest pain','claudication','palpitations','tachycardia','dry mouth','epistaxis','halitosis','hearing loss','nasal discharge','otalgia','otorrhea','sore throat','toothache','tinnitus','trismus','abdominal pain','fever','bloating','belching','bleeding','blood in stool','melena','hematochezia', 'constipation','diarrhea','dysphagia','dyspepsia','fecal incontinence','flatulence','heartburn','nausea','odynophagia','proctalgia fugax','pyrosis','steatorrhea','vomiting','alopecia','hirsutism','hypertrichosis','abrasion','anasarca','bleeding into the skin','petechia','purpura','ecchymosis and bruising','blister','edema','itching','laceration','rash','urticaria','abnormal posturing','acalculia','agnosia','alexia','amnesia','anomia','anosognosia','aphasia and apraxia','apraxia','ataxia','cataplexy','confusion','dysarthria','dysdiadochokinesia','dysgraphia','hallucination','headache','akinesia','bradykinesia','akathisia','athetosis','ballismus','blepharospasm','chorea','dystonia','fasciculation','muscle cramps','myoclonus','opsoclonus','tic','tremor','flapping tremor','insomnia','loss of consciousness','syncope','neck stiffness','opisthotonus','paralysis and paresis','paresthesia','prosopagnosia','somnolence','abnormal vaginal bleeding','vaginal bleeding in early pregnancy', 'miscarriage','vaginal bleeding in late pregnancy','amenorrhea','infertility','painful intercourse','pelvic pain','vaginal discharge','amaurosis fugax','amaurosis','blurred vision','double vision','exophthalmos','mydriasis','miosis','nystagmus','amusia','anhedonia','anxiety','apathy','confabulation','depression','delusion','euphoria','homicidal ideation','irritability','mania','paranoid ideation','suicidal ideation','apnea','hypopnea','cough','dyspnea','bradypnea','tachypnea','orthopnea','platypnea','trepopnea','hemoptysis','pleuritic chest pain','sputum production','arthralgia','back pain','sciatica','Urologic','dysuria','hematospermia','hematuria','impotence','polyuria','retrograde ejaculation','strangury','urethral discharge','urinary frequency','urinary incontinence','urinary retention']

organs = ['mouth','teeth','tongue','salivary glands','parotid glands','submandibular glands','sublingual glands','pharynx','esophagus','stomach','small intestine','duodenum','Jejunum','ileum','large intestine','liver','Gallbladder','mesentery','pancreas','anal canal and anus','blood cells','respiratory system','nasal cavity','pharynx','larynx','trachea','bronchi','lungs','diaphragm','Urinary system','kidneys','Ureter','bladder','Urethra','reproductive organs','ovaries','Fallopian tubes','Uterus','vagina','vulva','clitoris','placenta','testes','epididymis','vas deferens','seminal vesicles','prostate','bulbourethral glands','penis','scrotum','endocrine system','pituitary gland','pineal gland','thyroid gland','parathyroid glands','adrenal glands','pancreas','circulatory system','Heart','patent Foramen ovale','arteries','veins','capillaries','lymphatic system','lymphatic vessel','lymph node','bone marrow','thymus','spleen','tonsils','interstitium','nervous system','brain','cerebrum','cerebral hemispheres','diencephalon','the brainstem','midbrain','pons','medulla oblongata','cerebellum','the spinal cord','the ventricular system','choroid plexus','peripheral nervous system','nerves','cranial nerves','spinal nerves','Ganglia','enteric nervous system','sensory organs','eye','cornea','iris','ciliary body','lens','retina','ear','outer ear','earlobe','eardrum','middle ear','ossicles','inner ear','cochlea','vestibule of the ear','semicircular canals','olfactory epithelium','tongue','taste buds','integumentary system','mammary glands','skin','subcutaneous tissue']

higher_terms = ['over', 'above', 'higher', 'older', '>', 'over', 'less']

lower_terms = ['under', 'below', 'fewer', 'younger', '<', 'under', 'more']
matchers = {    

    "Term Matcher": lambda term: [{'LOWER': t} for t in term.split(' ')],

    "Terms Matcher": lambda terms: [{"LOWER": {"IN": terms } }],

    "Number Suffix Matcher": lambda periods: [

        {'LIKE_NUM': True},

        {"TEXT": {"REGEX": f'({"|".join(periods)})'}}

    ],

    "Number Interval Matcher": lambda periods: [

        {'POS': 'NUM',},

        {'TEXT': {'REGEX': f'({"|".join(periods)})'}, 'OP': '?'},

        {'DEP': 'quantmod', 'OP': '?'},

        {'DEP': 'punct', 'OP': '?'},

        {'DEP': 'prep', 'OP': '?'},

        {'POS': 'NUM'},

        {'TEXT': {'REGEX': f'({"|".join(periods)})'}},

    ],

    "Group Matcher": [

        {"TEXT": {"IN": higher_terms+lower_terms }}

    ]

}
def plot_dict(stat, t = 10, sort_values = False, barh = False, width = 20, height = 4, title = ''):

    filtered = dict(stat)

    to_delete = []

    for key in filtered:

        if filtered[key] < t:

            to_delete.append(key)

    for key in to_delete:

        del filtered[key]



    

    if sort_values == False:

        lists = sorted(filtered.items())

    else:

        if sort_values == True:

            lists = sorted(filtered.items(), key = lambda item : item[1])

        else:

            lists = sorted(filtered.items(), key = sort_values)

               

    fig = figure(num=None, figsize=(width, height))

    

    if title != '':

        fig.suptitle(title, fontsize=20)

        

    x, y = zip(*lists) 

    

    if barh == True:

        plt.barh(x, y)

    else:

        plt.bar(x, y)

    plt.show()

    



def merge_keys(mergers, obj):

    result = dict(obj)

    for key, arr in mergers:

        if key not in result:

            result[key] = 0

        for merger in arr:

            if merger in result:

                result[key] = result[key] + result[merger]

                del result[merger]

    return result



def dict_counter(res, arg):

    try:

        key = str(arg)

        res.setdefault(key, 0)

        res[key] = res[key] + 1

    except:

        pass



def numval(val):

    try:

        return int(float(str(val))) 

    except:

        return None

    

def day_value(val, rep = None):

    

    if rep != None:

        val = numval(val.text)

        if val != None and 'week' in rep.text:

            val = val * 7

        return val

    else:

        return None



def report_interval(res, min_val, max_val):       

    if min_val != None and max_val != None:

        for key in range(min_val, max_val):

            res.setdefault(key, 0)

            res[key] = res[key] + 1    



def virus_match(text):

    return len(re.findall(rf'({"|".join(virus_ref)})', text, flags=re.IGNORECASE)) > 0
literature = []

for index, row in tqdm(df.iterrows(), total=df.shape[0]):

    sha = str(row['sha'])

    if sha != 'nan':

        sha = sha + '.json';

        try:

            found = False

            with open(articles[sha]) as f:

                data = json.load(f)

                for key in ['abstract', 'body_text']:

                    if found == False and key in data:

                        for content in data[key]:

                            text = content['text']

                            if virus_match(text) == True:                                

                                literature.append({'file': articles[sha], 'body': text})                                

        except KeyError:

            pass
def execute_matches(match_arr, root, sentence, file, index = 0, execution = []):

    key, result = match_arr[0]

    rest = match_arr[1:]

    next_exec = execution + [(key, result, index)]

    if key in root:

        rule = root[key]

        if callable(rule):

            rule( (result, next_exec, sentence, file) )            

        else:

            if 'execute' in rule:

                rule['execute']( (result, next_exec, sentence, file) )

            if len(rest) > 0:

                execute_matches(rest, rule, sentence, file, index+1, next_exec)

    

    if len(rest) > 0:               

        execute_matches(rest, root, sentence, file, index + 1, execution)

        

def merge_dict_values(original, rules, drop = []):

    result = {}

    arr_map = {}

    for key, values in rules:

        for val in values:

            arr_map[val] = key

    

    for key in original.keys():

        new_key = key if key not in arr_map else arr_map[key]        

        if key not in drop and new_key not in drop:

            val = original[key]            

            result[new_key] = val if new_key not in result else result[new_key] + val

            

    return result

    

def merge_matches(matches, doc):

    match_list = []

    current = (None, None, None)

    for match_id, start, end in matches:   

        if match_id != current[0] or current[2] < start:

            if current[0] != None:

                match_list.append(current)

            current = (match_id, start, end)

        elif current[2] < end:

            current = (match_id, current[1], end)

        

    match_list.append(current)

    return match_list;



def match_parser(matcher, doc, rule, file):

    matches = matcher(doc)

    if len(matches)>0:

        to_process = []

        for match_id, start, end in merge_matches(matches, doc):

            string_id = nlp.vocab.strings[match_id]  # Get string representation

            span = doc[start:end]  # The matched span

            to_process.append((string_id, span))

        execute_matches(to_process, rule['root'], doc, file)



def parse_body(matcher, text, rule, file = None, sentence_level = False):

    text = text.lower()

    doc = nlp(text)

    

    if sentence_level == True:    

        for sent in doc.sents:

            sent_doc = nlp(sent.text)

            match_parser(matcher, sent_doc, rule, file)

    else:

        match_parser(matcher, doc, rule, file)



def execute_ruleset(term, rule, sentence_level = False, literature = literature):

    matcher = Matcher(nlp.vocab)

    for name, m in rule["Matchers"]:

        matcher.add(name, None, m)

    

    for article in tqdm(literature):

#     for article in literature:

        text_list = re.compile("\. ").split(article['body'])

        file = article['file']

        for text in text_list:

            if callable(term):

                allow = term(text)

            else:

                allow = term == None or term in text

            if allow == True:

                parse_body(matcher, text, rule, file, sentence_level)        
stat['symptoms'] = {}



def match(text):

    if virus_match(text) == True:

        return len(re.findall(rf'\ ({"|".join(symptoms)})\ ', text)) > 0

    else:

        return False



def symptom(res):

    ref, agregate, sentence, file = res

    dict_counter(stat['symptoms'], ref.text)

        

rule = {    

    "Matchers": [      

       ("Symptoms Reference", matchers['Terms Matcher'](symptoms)),

    ],

    "root": {

        "Symptoms Reference": symptom

    }

}





def symptom_match(text):

    return len(re.findall(r'symptom', text)) > 0



execute_ruleset(symptom_match, rule)

plot_dict(stat['symptoms'], 50, True, title = "Manifested Symptoms")
stat['incubation_periods'] = {}



def incubation_period_report(x):

    arr = x[1][-2:]

    m1, v1, i1 = arr[0]

    m2, v2, i2 = arr[1]

    

    if m1 == 'Term Matcher':

        if m2 == 'Time Matcher':

            report_interval(stat['incubation_periods'], 0, day_value(v2[0], v2[1]))            

        elif m2 == 'Time Interval Matcher':

            report_interval(stat['incubation_periods'], day_value(v2[0], v2[3]), day_value(v2[1], v2[1]))           

    elif m2 == 'Term Matcher':

        if m2 == 'Time Matcher':

            report_interval(stat['incubation_periods'], 0, day_value(v2[0], v2[1]))

        elif m2 == 'Time Interval Matcher':

            report_interval(stat['incubation_periods'], day_value(v2[0], v2[3]), day_value(v2[1], v2[1]))      



rule = {    

    "Matchers": [

        ("Term Matcher", [

            {"LOWER": "incubation"},

            {"LOWER": "period", "OP": "?"}

        ]),

        ("Time Matcher", matchers["Number Suffix Matcher"](["days", "weeks"])),

        ("Time Interval Matcher", matchers["Number Interval Matcher"](["days", "weeks"]))

    ],

    "root": {          

        "Term Matcher": { 

            "Time Matcher": incubation_period_report ,

            "Time Interval Matcher": incubation_period_report,

        },

        "Day Matcher": { "Term Matcher": incubation_period_report },

        "Day Interval Matcher": { "Term Matcher": incubation_period_report }

    }

}



execute_ruleset('incubation period', rule)

plot_dict(stat['incubation_periods'], 15, title = 'Incubation Period')
stat['quarantine'] = {}



def quarantine_matcher(text):

    return virus_match(text) == True and 'quarantine' in text



def quarantine_report(x):

    arr = x[1][-2:]

    m1, v1, i1 = arr[0]

    m2, v2, i2 = arr[1]

    

    if m1 == 'Quarantine Matcher':

        if m2 == 'Time Matcher':

            report_interval(stat['quarantine'], 0, day_value(v2[0], v2[1]))            

        elif m2 == 'Time Interval Matcher':

            report_interval(stat['quarantine'], day_value(v2[0], v2[3]), day_value(v2[1], v2[1]))           

    elif m2 == 'Quarantine Matcher':

        if m2 == 'Time Matcher':

            report_interval(stat['quarantine'], 0, day_value(v2[0], v2[1]))

        elif m2 == 'Time Interval Matcher':

            report_interval(stat['quarantine'], day_value(v2[0], v2[3]), day_value(v2[1], v2[1]))      

            

rule = {    

    "Matchers": [

        ("Quarantine Matcher", [

            {"LOWER": "quarantine"},

        ]),

        

        ("Time Matcher", matchers["Number Suffix Matcher"](["days", "weeks"])),

        ("Time Interval Matcher", matchers["Number Interval Matcher"](["days", "weeks"]))

    ],

    "root": {          

        "Quarantine Matcher": { 

            "Time Matcher": quarantine_report ,

            "Time Interval Matcher": quarantine_report,

        },

        "Day Matcher": { "Quarantine Matcher": quarantine_report },

        "Day Interval Matcher": { "Quarantine Matcher": quarantine_report }

    }

}



execute_ruleset('quarantine', rule)

plot_dict(stat['quarantine'], 10, title = 'Quarantine Period')
stat['transmission'] = {

}



def report_term(x):

    arr = x[1]

    m1, v1, i1 = arr[0]

    m2, v2, i2 = arr[1]

    m3, v3, i3 = arr[2]

    

    if m1 == 'Term Matcher' and m2 == 'Form Matcher':

        dict_counter(stat['transmission'], re.sub(r'[ ]?(-|the| a )[ ]?','', v3.text))



term_match = ['transmit','transmitted', 'spread', 'spreaded']

rule = {    

    "Matchers": [

        ("Term Matcher", [{"LOWER": "now", "OP": "!"}] + matchers["Terms Matcher"](term_match)),

        ("Form Matcher", matchers["Terms Matcher"](['through', 'by', 'via'])),

        ("Noun Matcher", [

            {"POS": "VERB", "OP": "?"},

            {"POS": "DET", "OP": "?"},

            {"POS": "ADJ", "OP": "?"},

            {"POS": "PUNCT", "OP": "?"},

            {"POS": "ADJ", "OP": "?"},

            {"POS": "NOUN", "OP": "+"},

        ])

    ],

    "root": {          

        "Term Matcher": {

            "Form Matcher": {

                "Noun Matcher": report_term

            }

        }

    }

}





def transmission_match(text):

    return len(re.findall(rf'({"|".join(term_match)})', text)) >0



execute_ruleset(transmission_match, rule, False)



new_dict = merge_dict_values(stat['transmission'], [

    ('contact', ['direct contact', 'close contact', 'indirect contact', 'person contact']),

    ('respiratory droplets/route', ['droplets','air', 'airborne route', 'aerosols', 'airborne transmission', 'respiratory route','respiratory droplets', 'droplet', 'respiratory secretions']),

    ('surfaces/fomites', ['fomites', 'surfaces', 'environmental surfaces', 'environment']),

    ('human transmission', ['human','humans','patient', 'patients','person', 'people']),

    ('fecal-oral route', ['fecaloral route', 'faecaloral route'])

], ['%', 'virus', 'viruses'])





plot_dict(new_dict, 30, True, barh = True, height = 10, title = 'Transmission Routes')
stat['organs'] = {}



def match(text):

    if virus_match(text) == True:

        return len(re.findall(rf'\ ({"|".join(organs)})\ ', text)) > 0

    else:

        return False



def organ_reference(res):

    ref, agregate, sentence, file = res

    dict_counter(stat['organs'], ref.text)

        

rule = {    

    "Matchers": [      

       ("Organ Reference", matchers['Terms Matcher'](organs)),

    ],

    "root": {

        "Organ Reference": organ_reference

    }

}





def symptom_match(text):

    res = re.findall(rf'\ ({"|".join(organs)})\ ', text, flags=re.IGNORECASE)    

    return len(res) >0



execute_ruleset(symptom_match, rule)

plot_dict(stat['organs'], 500, True, title = 'Affected Organs')
stat['gender'] = {

    'male': {},

    'female': {}

}



resolution = 10

count = int(100 / resolution)

for val in range(0, count):

    key = f'{val*resolution}-{val*resolution+resolution}'

    stat['gender']['male'][key] = 0

    stat['gender']['female'][key] = 0



def get_key(gender):

    return 'female' if 'female' in gender else 'male'

    

def percent_counter(x):

    arr = x[1][-2:]

    m1, v1, i1 = arr[0]

    m2, v2, i2 = arr[1]

   

    if m1 != m2 and i1 == i2-1:

        gender = get_key(v1.text if m1 == 'Gender Matcher' else v2.text)

        percent = numval((v2 if m1 == 'Gender Matcher' else v1)[0])

        if percent != None and percent >= 0 and percent <=100:

            percent = int(percent / resolution) * resolution

            dict_counter(stat['gender'][gender], f'{percent}-{percent+resolution}')



rule = {    

    "Matchers": [

        ("Gender Matcher", matchers["Terms Matcher"](["male", "males", "female", "females"])),

        ("Percent Matcher", matchers["Number Suffix Matcher"](["%", "percent"])),

    ],

    "root": {

        "Gender Matcher": { "Percent Matcher":  percent_counter },

        "Percent Matcher": { "Gender Matcher":  percent_counter },

    }

}



def gender_match(text):

    return len(re.findall(rf'(male|female)', text, flags=re.IGNORECASE)) > 0





execute_ruleset(gender_match, rule, False)



final_arr = []



for i, key in enumerate(stat['gender']['male'].keys()):

    final_arr.append([i, key, stat['gender']['male'][key], 'male'])

    

for i, key in enumerate(stat['gender']['female'].keys()):

    final_arr.append([i, key, stat['gender']['female'][key], 'female'])

    

df = pd.DataFrame(final_arr, columns = ['index', 'range', 'count', 'gender'])

sns.barplot(x="range", y="count", hue="gender", data=df)

stat['fatality'] = {}



    

def percent_counter(x):

    arr = x[1][-2:]

    m1, v1, i1 = arr[0]

    m2, v2, i2 = arr[1]

    val = numval(v2[0])

    if val != None and m1 == 'Fatality Matcher' and i1 == i2 - 1:

        dict_counter(stat['fatality'], val)   



rule = {    

    "Matchers": [

        ("Fatality Matcher", matchers["Terms Matcher"](["mortality", "fatality"]) + [{"LOWER": "rate"}]),

        ("Percent Matcher", matchers["Number Suffix Matcher"](["%", "percent"])),

    ],

    "root": {

        "Fatality Matcher": {

            "Percent Matcher":  percent_counter

        }

    }

}



def fatality_match(text):

    return len(re.findall(rf'(mortality|fatality) rate', text, flags=re.IGNORECASE)) > 0





execute_ruleset(fatality_match, rule, False)

plot_dict(stat['fatality'], 25, sort_values = lambda item : float(item[0]), title="Fatality Rate Reports")
stat['genome'] = []



regex = r' ([GTCA]{2,}[GTCA\-\~\ ]{3,}[GTCA])\W'

def sequence_matcher(x):

    text, match, sent, file = x

    name, found, index = match[0]

    matches = re.finditer(regex, sent.text, re.MULTILINE | re.IGNORECASE)

    genome = [match.group(0).strip() for matchNum, match in enumerate(matches, start=1)]

    stat['genome'].append({'genome': genome, 'sentence': sent.text, 'file': file})



rule = {    

    "Matchers": [

        ('Genome Matcher', [

            {"lower": {"regex": regex}}

        ])

    ],

    "root": {

        "Genome Matcher": sequence_matcher

    }

}



def fatality_match(text):

    return len(re.findall(regex, text, flags=re.IGNORECASE)) > 0



execute_ruleset(fatality_match, rule, False)

pd.DataFrame(stat['genome']) 
stat['false_pos_neg'] = {

    'words': {},

    'refs': []

}



# regex = r' °[CF] '

regex = r' false [negative|positive]'



exclude = ['false', 'positive', 'positives', 'negative', 'negatives', 'value', 'values', 'number', 'use', 'fig', 'site'] + virus_ref

def fp_matcher(x):

    text, match, sent, file = x

    name, found, index = match[0]

    for token in sent:

        if token.pos_ in ['NOUN'] and token.is_punct == False and token.is_stop == False and token.text not in exclude:

            dict_counter(stat['false_pos_neg']['words'], token.text)

    stat['false_pos_neg']['refs'].append({'sentence': sent.text, 'file': file})



rule = {    

     "Matchers": [

        ('Term Matcher', [

            {"lower": {"regex": regex}}

        ])

    ],

    "root": {

        "Term Matcher": fp_matcher

    }

}



def regex_match(text):

    return len(re.findall(regex, text, flags=re.IGNORECASE)) > 0



execute_ruleset(regex_match, rule, False)

stat['false_pos_neg']['refs'] = pd.DataFrame(stat['false_pos_neg']['refs'])

stat['false_pos_neg']['refs']
new_dict = merge_dict_values(stat['false_pos_neg']['words'], [

    ('tesst', ['testing', 'test']),

    ('viruses', ['virus']),

    ('values', ['value']),

    ('studies', ['study']),

    ('tests', ['test', 'tested']),

    ('signales', ['signal']),

    ('antibodies', ['antibody']),

    ('specificity', ['specific'])

])

plot_dict(new_dict, 5, sort_values = True, title="False Positive/Negative Terms", barh = True, height = 20,)
stat['antigen'] = {

    'count': {},

    'refs': []

}



# regex = r' (antigen|antibod)'

regex = r' antigen'

exclude = []



def antigen_matcher(x):

    text, match, sent, file = x

    

    if text.text != 'antigen':

#         text = text.text.replace(.replace('antigens', '').replace('antigen', '').strip()        

        text = re.sub(r'(antibodies|antibody|antigens|antigen)', '', text.text).strip()

        if len(text) > 0:

            dict_counter(stat['antigen']['count'], text)

            stat['antigen']['refs'].append({'sentence': sent.text, 'file': file})



rule = {    

     "Matchers": [

        ('Term Matcher', [

            {"POS": "PROPN", "op": "*"},

            {"POS": "NOUN", "op": "*"},

            {"POS": "VERB", "op": "*"},

            {"LOWER": {"IN": ['antigen', 'antigens'] }} #, 'antibody', 'antibodies'] } }

        ])

    ],

    "root": {

        "Term Matcher": antigen_matcher

    }

}



def regex_match(text):

    return len(re.findall(regex, text, flags=re.IGNORECASE)) > 0



execute_ruleset(regex_match, rule, False)

stat['antigen']['refs'] = pd.DataFrame(stat['antigen']['refs'])

stat['antigen']['refs']
new_dict = merge_dict_values(

    stat['antigen']['count'], 

    [],

    ['coronavirus', 'neutralizing', 'neutralising', 'hcv', 'cov', 'hcv core', 'hcv'])

plot_dict(new_dict, 5, sort_values = True, title="Antigen References", barh = True, height = 20,)