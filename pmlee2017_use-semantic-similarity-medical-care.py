# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import matplotlib.pylab as plt

from IPython.display import display, HTML

from tabulate import tabulate

import query_covid19 as qc

FILEDIR = os.path.join("../input/","use-covid19-search-engine")



# Any results you write to the current directory are saved as output.
df_meta = pd.read_csv(os.path.join(FILEDIR, "df_meta_comp.csv" )).set_index("row_id")

embs_title = np.load(os.path.join(FILEDIR, "embeddings_titles.npy" ))

embs_title_abstract = np.load(os.path.join(FILEDIR, "embeddings_title_abstract.npy" ))



# Initialize

qcovid = qc.qCOVID19(df_meta, embs_title, embs_title_abstract)
# define function for better display

def display_q(text, df_top, display_cols = ["publish_date", "title", "shorten_abstract", "authors", "journal", "similarities"]):

    display(HTML("Search term : <b>%s<b>" %text))

    display(HTML(tabulate(df_top[display_cols], headers = display_cols,tablefmt='html')))
questions_temp = [

    "Resources to support skilled nursing facilities and long term care facilities.",

    "Mobilization of surge medical staff to address shortages in overwhelmed communities",

    "Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies",

    "Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients",

    "Outcomes data for COVID-19 after mechanical ventilation adjusted for age.",

    "Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.",

    "Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.",

    "Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.",

    "Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.",

    "Guidance on the simple things people can do at home to take care of sick people and manage disease.",

    "Oral medications that might potentially work.",

    "Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually.",

    "Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcomes.",

    "Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials",

    "Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials",

    "Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients e.g. steroids, high flow oxygen"

]



#prepend these words, to emphasize on covid19

prepend_words = ["Covid-19, coronavirus"]

questions = []



for question in questions_temp :

    questions.append(", ".join(prepend_words + [question] ))
def generate_resp(question, ntop=10):

    df = qcovid.query(question, abstract_width=200)

    display_q(question, df.head(ntop).sort_values(["publish_date"], ascending=False))

    fig, ax = plt.subplots(figsize=(6, 4))

    qcovid.word_cloud(df, ax)

    

    
generate_resp(questions[0])

    
generate_resp(questions[1])
generate_resp(questions[2])
generate_resp(questions[3])
generate_resp(questions[4])
generate_resp(questions[5])
generate_resp(questions[6])
generate_resp(questions[7])
generate_resp(questions[8])
generate_resp(questions[9])
generate_resp(questions[10])
generate_resp(questions[11])
generate_resp(questions[12])
generate_resp(questions[13])
generate_resp(questions[14])