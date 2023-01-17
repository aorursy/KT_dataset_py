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

    "Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.",

    "Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.",

    "Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.",

    "Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.",

    "Experimental infections to test host range for this pathogen.",

    "Animal hosts and any evidence of continued spill-over to humans",

    "Socioeconomic and behavioral risk factors for this spill-over",

    "Sustainable risk reduction strategies",

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