# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        with open(os.path.join(dirname, filename)) as f:

            line_count = 0

            id_set = set()

            for l in f.readlines():

                line_count += 1

                if filename == "CISI.REL":

                    id_set.add(l.lstrip(" ").split(" ")[0])

                elif l.startswith(".I "):

                    id_set.add(l.split(" ")[1].strip())

            print(f"{filename} : {len(id_set)} items, over {line_count} lines.")



# Any results you write to the current directory are saved as output.
with open('/kaggle/input/cisi-a-dataset-for-information-retrieval/CISI.ALL') as f:

    lines = ""

    for l in f.readlines():

        lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()

    lines = lines.lstrip("\n").split("\n")

    

# print n lines

n = 5

for l in lines[:n]:

    print(l)
doc_set = {}

doc_id = ""

doc_text = ""

for l in lines:

    if l.startswith(".I"):

        doc_id = l.split(" ")[1].strip()

    elif l.startswith(".X"):

        doc_set[doc_id] = doc_text.lstrip(" ")

        doc_id = ""

        doc_text = ""

    else:

        doc_text += l.strip()[3:] + " " # The first 3 characters of a line can be ignored.



# Print something to see the dictionary structure, etc.

print(f"Number of documents = {len(doc_set)}" + ".\n")

print(doc_set["3"]) # note that the dictionary indexes are strings, not numbers. 

        
with open('/kaggle/input/cisi-a-dataset-for-information-retrieval/CISI.QRY') as f:

    lines = ""

    for l in f.readlines():

        lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()

    lines = lines.lstrip("\n").split("\n")

    

qry_set = {}

qry_id = ""

for l in merged.split("\n"):

    if l.startswith(".I"):

        qry_id = l.split(" ")[1].strip()

    elif l.startswith(".W"):

        qry_set[qry_id] = l.strip()[3:]

        qry_id = ""

    

# Print something to see the dictionary structure, etc.

print(f"Number of queries = {len(qry_set)}" + ".\n")

print(qry_set["3"]) # note that the dictionary indexes are strings, not numbers. 
rel_set = {}

with open('/kaggle/input/cisi-a-dataset-for-information-retrieval/CISI.REL') as f:

    for l in f.readlines():

        qry_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]

        doc_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1]

        if qry_id in rel_set:

            rel_set[qry_id].append(doc_id)

        else:

            rel_set[qry_id] = []

            rel_set[qry_id].append(doc_id)

        if qry_id == "7":

            print(l.strip("\n"))

    

# Print something to see the dictionary structure, etc.

print(f"\nNumber of mappings = {len(rel_set)}" + ".\n")

print(rel_set["7"]) # note that the dictionary indexes are strings, not numbers. 