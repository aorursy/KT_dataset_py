import json

import os

import pandas as pd

import re

import statistics

import matplotlib.pyplot as plt

from matplotlib import style

import numpy as np



style.use("ggplot")
def get_days_distribution(dir):

    docs = []

    for file in os.listdir(f"{dir}"):

        file_path = f"{dir}/{file}"

        j = json.load(open(file_path, "rb"))

        title = j['metadata']['title']

        try:

            abstract = j['abstract'][0]

        except:

            abstract = ""



        full_text = ""

        for text in j['body_text']:

            full_text += text['text'] + '\n\n'



        docs.append([title, abstract, full_text])



    df = pd.DataFrame(docs, columns=['title', 'abstract', 'full_text'])

    df.head()



    incubation = df[df['full_text'].str.contains('incubation')]



    texts = incubation['full_text'].values

    len(texts)



    incubation_times = []



    for t in texts:

        for sentence in t.split('. '):

            if "incubation" in sentence:

                single_day = re.findall(r'( \d{1,2}(\.\d{1,2})? day[s]?)', sentence)

                if len(single_day) == 1:

                    num = str(single_day[0]).split(" ")

                    incubation_times.append(float(num[1]))

    print("Number of sentences containing the word incubation: ", len(incubation_times))

    return(incubation_times)
def show_days_distribution(days):

    print(f"Average days : {statistics.mean(days)} days")

    with plt.style.context('dark_background'):

        plt.hist(days, int(max(days)))

        plt.xlabel("Number of Days")

        plt.ylabel("Distribution of Incubation days")

    plt.show()
biorxiv_medrxiv = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'

filenames = os.listdir(biorxiv_medrxiv)

print("Number of articles retrieved from biorxiv_medrxiv:", len(filenames))

biorxiv_medrxiv_dist = get_days_distribution(biorxiv_medrxiv)

show_days_distribution(biorxiv_medrxiv_dist)
comm_use_subset = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'

filenames = os.listdir(comm_use_subset)

print("Number of articles retrieved from comm_use_subset:", len(filenames))

comm_use_subset_dist = get_days_distribution(comm_use_subset)

show_days_distribution(comm_use_subset_dist)
custom_license = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'

filenames = os.listdir(custom_license)

print("Number of articles retrieved from custom_license:", len(filenames))

custom_license_dist = get_days_distribution(custom_license)

show_days_distribution(custom_license_dist)
noncomm_use_subset = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'

filenames = os.listdir(noncomm_use_subset)

print("Number of articles retrieved from noncomm_use_subset:", len(filenames))

noncomm_use_subset_dist = get_days_distribution(noncomm_use_subset)

show_days_distribution(noncomm_use_subset_dist)
cumulative_dist = biorxiv_medrxiv_dist + comm_use_subset_dist + custom_license_dist + noncomm_use_subset_dist

show_days_distribution(cumulative_dist)

# statistics.mean(cumulative_dist)