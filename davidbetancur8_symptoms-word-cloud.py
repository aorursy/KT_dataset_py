import os

import re

import json

from tqdm import tqdm

import pandas as pd

from collections import Counter

from stop_words import get_stop_words

from wordcloud import WordCloud

import matplotlib.pyplot as plt
cord_path = '../input/CORD-19-research-challenge'

dirs = ["biorxiv_medrxiv", "comm_use_subset", "noncomm_use_subset", "custom_license"]
docs = []

for d in dirs:

    for file in tqdm(os.listdir(f"{cord_path}/{d}/{d}")):

        file_path = f"{cord_path}/{d}/{d}/{file}"

        j = json.load(open(file_path, "rb"))



        title = j["metadata"]["title"]

        authors = j["metadata"]["authors"]



        try:

            abstract = j["abstract"][0]["text"].lower()

        except:

            abstract = ""



        full_text = ""

        for text in j["body_text"]:

            full_text += text["text"].lower() + "\n\n"

        docs.append([title, authors, abstract, full_text])



df = pd.DataFrame(docs, columns=["title", "authors", "abstract", "full_text"])
symptoms_df = df[df["full_text"].str.contains("symptom")]
symptoms = [

    "weight loss","chills","shivering","convulsions","deformity","discharge","dizziness",

    "vertigo","fatigue","malaise","asthenia","hypothermia","jaundice","muscle weakness",

    "pyrexia","sweats","swelling","swollen","painful lymph node","weight gain","arrhythmia",

    "bradycardia","chest pain","claudication","palpitations","tachycardia","dry mouth","epistaxis",

    "halitosis","hearing loss","nasal discharge","otalgia","otorrhea","sore throat","toothache","tinnitus",

    "trismus","abdominal pain","fever","bloating","belching","bleeding","blood in stool","melena","hematochezia",

    "constipation","diarrhea","dysphagia","dyspepsia","fecal incontinence","flatulence","heartburn",

    "nausea","odynophagia","proctalgia fugax","pyrosis","steatorrhea","vomiting","alopecia","hirsutism",

    "hypertrichosis","abrasion","anasarca","bleeding into the skin","petechia","purpura","ecchymosis and bruising",

    "blister","edema","itching","laceration","rash","urticaria","abnormal posturing","acalculia","agnosia","alexia",

    "amnesia","anomia","anosognosia","aphasia and apraxia","apraxia","ataxia","cataplexy","confusion","dysarthria",

    "dysdiadochokinesia","dysgraphia","hallucination","headache","akinesia","bradykinesia","akathisia","athetosis",

    "ballismus","blepharospasm","chorea","dystonia","fasciculation","muscle cramps","myoclonus","opsoclonus","tic",

    "tremor","flapping tremor","insomnia","loss of consciousness","syncope","neck stiffness","opisthotonus",

    "paralysis and paresis","paresthesia","prosopagnosia","somnolence","abnormal vaginal bleeding",

    "vaginal bleeding in early pregnancy", "miscarriage","vaginal bleeding in late pregnancy","amenorrhea",

    "infertility","painful intercourse","pelvic pain","vaginal discharge","amaurosis fugax","amaurosis",

    "blurred vision","double vision","exophthalmos","mydriasis","miosis","nystagmus","amusia","anhedonia",

    "anxiety","apathy","confabulation","depression","delusion","euphoria","homicidal ideation","irritability",

    "mania","paranoid ideation","suicidal ideation","apnea","hypopnea","cough","dyspnea","bradypnea","tachypnea",

    "orthopnea","platypnea","trepopnea","hemoptysis","pleuritic chest pain","sputum production","arthralgia",

    "back pain","sciatica","Urologic","dysuria","hematospermia","hematuria","impotence","polyuria",

    "retrograde ejaculation","strangury","urethral discharge","urinary frequency","urinary incontinence","urinary retention"]
texts = df.full_text.values



all_words = []

for text in texts:

    sentences = re.split('[. ] |\n',text)

    for sentence in sentences:

        sentence = sentence.replace(',', '')

        if ("symptom" in sentence):

            words = sentence.split()

            words = [word for word in words if word  in symptoms]

            all_words.append(words)

            

all_words = [item for sublist in all_words for item in sublist]
word_dict = Counter(all_words)
wc = WordCloud(background_color="black",width=1000, height=800).generate_from_frequencies(word_dict)

fig = plt.figure(figsize=(15,15))

plt.imshow(wc, interpolation="bilinear")

plt.axis("off")

plt.show()