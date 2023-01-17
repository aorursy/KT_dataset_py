# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import json

import pandas as pd

from tqdm import tqdm # to make a progress bar

import re #hello regular expressions
docs = []

dirs = ["biorxiv_medrxiv", "comm_use_subset", "custom_license", "noncomm_use_subset"]



for d in dirs:

    print ("directory is: ", d)

    for file in tqdm(os.listdir(f"/kaggle/input/CORD-19-research-challenge/{d}/{d}")):

        file_path = f"/kaggle/input/CORD-19-research-challenge/{d}/{d}/{file}"

        

        j = json.load(open(file_path, "rb"))

        #print(j)

        title = j["metadata"]["title"]

        try:

            abstract = j["abstract"][0]["text"]

        except:

            abstract = ""

            #print(j["abstract"])

        

        #print(abstract)

        full_text = ""

        for text in j["body_text"]:

            #print(text['text'])

            full_text+=text['text']+"\n\n"

        #print(full_text)

        docs.append([title, abstract, full_text])

        #break

df = pd.DataFrame(docs, columns=["title", "abstract", "full_text"])

#print (df.head())

df_incubation = df[df["full_text"].str.contains("incubation")]

#print (df_incubation.head())

texts = df_incubation["full_text"].values



incubation_times = []
regx_to_search = [" \d{1,2}\.?\d{1,2} (?=day)" ,  # 14 days or 14.23 days

                  " \d{1,2}\.?\d{1,2} to \d{1,2}\.?\d{1,2} (?=day)",  #3 to 5 days

                  " \d{1,2}\.?\d{1,2}-\d{1,2}\.?\d{1,2} (?=day)"]   #3-5 days

            

for t in texts:

    #print(t)

    for sentence in t.split(". "):

        if "incubation" in sentence:

            #print(sentence, "\n\n")

            #single_day = re.findall(r" \d{1,2} day", sentence)

            for reg in regx_to_search:

                single_day = re.findall(reg, sentence)

                ## ?: 0 or 1 occurrence

                if len(single_day) >= 1: #so there is at least one incubation period in this sentence

                #if True:

                    #print("\n************\n","found: ",single_day, "\n**********************")

                    #print(sentence+".", "\n\n")

                    incubation_times.append(single_day)

#print (incubation_times)

final_incubation_times = []



import statistics



for entry in incubation_times:

    for p in entry:

        if "-" in p:

            divided = p.split("-")

            floated = [float(i) for i in divided]

            final_incubation_times.append(statistics.mean(floated))

        elif "to" in p:

            divided = p.split("to")

            floated = [float(i) for i in divided]

            final_incubation_times.append(statistics.mean(floated))

        else:

            final_incubation_times.append(float(p))

print(final_incubation_times[:50]) 

print (len(final_incubation_times))
file = open("incubation_data.txt","w+")

for time in final_incubation_times:

    file.write(str(time))

file.close()
file = open("incubation_data.txt","r")

if file.mode == 'r':

    contents = file.read()

file.close()
import matplotlib.pyplot as plt

from matplotlib import style

style.use("ggplot")



plt.hist(final_incubation_times, bins=1000)

plt.ylabel("bin counts")

plt.xlabel("incubation time (days)")

plt.xlim(0, 34)

plt.show()
import numpy as np

print(f"The mean projected incubation time is {np.mean([i for i in final_incubation_times if i<34])} days")