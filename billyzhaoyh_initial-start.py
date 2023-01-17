import numpy as np

import pandas as pd



import os

import json
count = 0

file_exts = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        count += 1

        file_ext = filename.split(".")[-1]

        file_exts.append(file_ext)



file_ext_set = set(file_exts)



print(f"Files: {count}")

print(f"Files extensions: {file_ext_set}\n\n=====================\nFiles extension count:\n=====================")

file_ext_list = list(file_ext_set)

for fe in file_ext_list:

    fe_count = file_exts.count(fe)

    print(f"{fe}: {fe_count}")
count = 0

for root, folders, filenames in os.walk('/kaggle/input'):

    print(root, folders)
json_folder_path = "/kaggle/input/CORD-19-research-challenge/2020-03-13/pmc_custom_license/pmc_custom_license"

json_file_name = os.listdir(json_folder_path)[0]

json_path = os.path.join(json_folder_path, json_file_name)



with open(json_path) as json_file:

    json_data = json.load(json_file)
json_data_df = pd.io.json.json_normalize(json_data)
json_data_df