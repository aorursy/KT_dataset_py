# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns



import re

import json

import math



import spacy

from spacy.matcher import Matcher



from tqdm import tqdm





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

nlp = spacy.load("en_core_web_sm")



articles = {}

stat = { }

# Import json files

for dirpath, subdirs, files in os.walk('/kaggle/input'):

    for i in files:

        if i.endswith(".json"):

            articles[i] = os.path.join(dirpath, i)

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')



symptoms = ['weight loss','chills','shivering','convulsions','deformity','discharge','dizziness','vertigo','fatigue','malaise','asthenia','hypothermia','jaundice','muscle weakness','pyrexia','sweats','swelling','swollen','painful lymph node','weight gain','arrhythmia','bradycardia','chest pain','claudication','palpitations','tachycardia','dry mouth','epistaxis','halitosis','hearing loss','nasal discharge','otalgia','otorrhea','sore throat','toothache','tinnitus','trismus','abdominal pain','fever','bloating','belching','bleeding','blood in stool','melena','hematochezia', 'constipation','diarrhea','dysphagia','dyspepsia','fecal incontinence','flatulence','heartburn','nausea','odynophagia','proctalgia fugax','pyrosis','steatorrhea','vomiting','alopecia','hirsutism','hypertrichosis','abrasion','anasarca','bleeding into the skin','petechia','purpura','ecchymosis and bruising','blister','edema','itching','laceration','rash','urticaria','abnormal posturing','acalculia','agnosia','alexia','amnesia','anomia','anosognosia','aphasia and apraxia','apraxia','ataxia','cataplexy','confusion','dysarthria','dysdiadochokinesia','dysgraphia','hallucination','headache','akinesia','bradykinesia','akathisia','athetosis','ballismus','blepharospasm','chorea','dystonia','fasciculation','muscle cramps','myoclonus','opsoclonus','tic','tremor','flapping tremor','insomnia','loss of consciousness','syncope','neck stiffness','opisthotonus','paralysis and paresis','paresthesia','prosopagnosia','somnolence','abnormal vaginal bleeding','vaginal bleeding in early pregnancy', 'miscarriage','vaginal bleeding in late pregnancy','amenorrhea','infertility','painful intercourse','pelvic pain','vaginal discharge','amaurosis fugax','amaurosis','blurred vision','double vision','exophthalmos','mydriasis','miosis','nystagmus','amusia','anhedonia','anxiety','apathy','confabulation','depression','delusion','euphoria','homicidal ideation','irritability','mania','paranoid ideation','suicidal ideation','apnea','hypopnea','cough','dyspnea','bradypnea','tachypnea','orthopnea','platypnea','trepopnea','hemoptysis','pleuritic chest pain','sputum production','arthralgia','back pain','sciatica','Urologic','dysuria','hematospermia','hematuria','impotence','polyuria','retrograde ejaculation','strangury','urethral discharge','urinary frequency','urinary incontinence','urinary retention']

higher_terms = ['over', 'above', 'higher', 'older', '>', 'over', 'less']

lower_terms = ['under', 'below', 'fewer', 'younger', '<', 'under', 'more']

virus_ref = ['covid-19', 'coronavirus', 'cov-2', 'sars-cov-2', 'sars-cov', 'hcov', '2019-ncov']
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

plot_dict(stat['symptoms'], 50, True, title = "Symptoms")
print(stat['symptoms'])
from wordcloud import WordCloud



    



wc = WordCloud(background_color="black",width=1000,height=1000, max_words=20,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(stat['symptoms'])

plt.figure(figsize=(10,7))

plt.imshow(wc, interpolation = "bilinear")

plt.axis('off')

plt.show()
pd.pandas.set_option('display.max_columns',None)

pd.pandas.set_option('display.max_rows',None)

symptoms_checker = pd.read_csv('/kaggle/input/covid19-symptoms-checker/Cleaned-Data.csv')

from wordcloud import WordCloud, STOPWORDS

# Create and generate a word cloud image:



#wordcloud = WordCloud(width=480, height=480,margin=0,stopwords=STOPWORDS, collocations=False).generate(' '.join(symptoms_checker))

#plt.figure(figsize=(20,10), facecolor='k')

#plt.imshow(wordcloud, interpolation='bilinear')

#plt.axis("off")

#plt.margins(x=0, y=0)

#plt.show()

#symptoms_checker.head()

data0 = symptoms_checker.drop("Country",1)

data0.apply(pd.value_counts).plot(y=["Fever","Tiredness", "Dry-Cough", "Difficulty-in-Breathing", "Sore-Throat", "None_Sympton", "Pains", "Nasal-Congestion", "Runny-Nose", "Diarrhea"],kind='bar',title='all types')
data = symptoms_checker.copy()

data = data.drop(['Severity_None','None_Sympton','None_Experiencing','Contact_Dont-Know','Country','Contact_No'],axis = 1)

data.head()
data1 = data.copy()

data1 = data.drop(['Severity_Moderate','Severity_Mild'],axis = 1)

y_data = data1['Severity_Severe']

x_data = data1.drop(['Severity_Severe'],axis = 1)
SEED = 42

from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val = train_test_split(x_data,y_data,test_size = 0.3,random_state = SEED)

X_train.head()
wordcloud1 = X_train.drop(['Age_0-9','Age_10-19','Age_20-24','Age_25-59','Age_60+','Gender_Female','Gender_Male','Gender_Transgender','Contact_Yes'],axis = 1)

wordcloud = WordCloud(width=480, height=480,margin=0,stopwords=STOPWORDS, collocations=False).generate(' '.join(wordcloud1))

plt.figure(figsize=(20,10), facecolor='k')

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()

symptoms_checker.head()



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.model_selection import GridSearchCV

from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf.fit(X_train, Y_train)





y_pred_train = clf.predict(X_train)

y_pred_val = clf.predict(X_val)

# rf.fit(X_train,Y_train)
from sklearn.metrics import confusion_matrix

confusion_matrix(Y_val,y_pred_val)
scoring = 'accuracy'

score = cross_val_score(clf,X_val,Y_val,cv = k_fold,n_jobs=1,scoring=scoring)

a = 1

for i in score:

    print("(",a,") ",i)

    a+=1

type(score)
score.mean()