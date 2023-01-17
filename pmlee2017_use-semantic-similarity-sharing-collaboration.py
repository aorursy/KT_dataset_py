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

    "Methods for coordinating data-gathering with standardized nomenclature.",

    "Sharing response information among planners, providers, and others.",

    "Understanding and mitigating barriers to information-sharing.",

    "How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).",

    "Integration of federal/state/local public health surveillance systems.",

    "Value of investments in baseline public health response infrastructure preparedness",

    "Modes of communicating with target high-risk populations (elderly, health care workers).",

    "Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).",

    "Communication that indicates potential risk of disease to all population groups.",

    "Misunderstanding around containment and mitigation.",

    "Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.",

    "Measures to reach marginalized and disadvantaged populations, underrepresented minorities",

    "Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.",

    "Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care"

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