import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import glob

import os

import json
TRAIN_DIR = "/kaggle/input/dfdc-input/dfdc_train_meta/dfdc_train_part_*"

PATH_JSON = '/kaggle/working/merged_metadata.json'
def get_meta_from_json(path):

    df = pd.read_json(path)

    df = df.T

    return df
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    return(np.transpose(tt))
all_json =  glob.glob(os.path.join(TRAIN_DIR, 'metadata.json'))

for i in range(0,49):

    if  '/kaggle/input/dfdc-input/dfdc_train_meta/dfdc_train_part_'+str(i)+'/metadata.json' not in all_json:

        print("Missing: "+str(i))

all_metadata = {}

for f in all_json:

    with open(f, "r") as infile:

        all_metadata.update(json.load(infile))

with open(PATH_JSON, "w") as outfile:

     json.dump(all_metadata, outfile)
meta_train_df = get_meta_from_json(PATH_JSON)

meta_train_df.head()
missing_data(meta_train_df)
unique_values(meta_train_df)
# now, for each unique value, how many deep fakes do we have?

number_of_fakes = meta_train_df['original'].value_counts()
sns.distplot(number_of_fakes.values,norm_hist=True)