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

    "articulate and translate existing ethical principles and standards to salient issues in COVID-2019",

    "embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight",

    "support sustained education, access, and capacity building in the area of ethics",

    "establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.",

    "develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)",

    "identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.",

    "identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.",

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