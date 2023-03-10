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

    "How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).",

    "Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.",

    "Recruitment, support, and coordination of local expertise and capacity (public, private???commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.",

    "National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).",

    "Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.",

    "Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).",

    "Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.",

    "Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.",

    "Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.",

    "Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.",

    "Policies and protocols for screening and testing.",

    "Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.",

    "Technology roadmap for diagnostics.",

    "Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.",

    "New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.",

    "Coupling genomics and diagnostic testing on a large scale.",

    "Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.",

    "Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.",

    "One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors."

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
generate_resp(questions[15])
generate_resp(questions[16])
generate_resp(questions[17])
generate_resp(questions[18])